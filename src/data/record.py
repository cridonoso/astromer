import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import polars as pl
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# from joblib import wrap_non_picklable_objects
from joblib import Parallel, delayed
# from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
import logging
import toml
from typing import List, Dict, Any


# Set up logging configuration
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def parse_dtype(value):
    if type(value) == int:
        return _int64_feature([value])
    if type(value) == float:
        return _float_feature([value])
    if type(value) == str:
        return _bytes_feature([str(value).encode()])
    raise ValueError('[ERROR] {} with type {} could not be parsed. Please use <str>, <int>, or <float>'.format(value, dtype(value)))

def substract_frames(frame1, frame2, on):
    frame1 = frame1[~frame1[on].isin(frame2[on])]
    return frame1

def write_config(context_features: List[str], sequential_features: List[str], config_path: str) -> None:
    """
    Writes the configuration to a toml file.

    Args:
        context_features (list): List of context features.
        sequential_features (list): List of sequential features.
        config_path (str): Path to the output config.toml file.
    """
    config = {
        "context_features": context_features,
        "sequential_features": sequential_features
    }

    # Make directory if it does not exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    try:
        with open(config_path, 'w') as config_file:
            toml.dump(config, config_file)
        logging.info(f'Successfully wrote config to {config_path}')
    except Exception as e:
        logging.error(f'Error while writing the config file: {str(e)}')
        raise e


class DataPipeline:
    """
    Args:
            sequential_features (list[str]): seuqential features keys, order important, [0] : mjd, [1] : mag 
            context_features (list[str]): Context feature keys
            id_key (str) : id key for paraquet files
            err_threshold (int) : used to clean up dataset by error 
    """

    def __init__(self,
                 metadata=None,
                 config_path= "./config.toml", 
                 output_folder='./records/output', 
                 id_key = 'newID', 
                 err_threshold=1, ):

        self.metadata             = metadata
        self.output_folder        = output_folder
        self.id_key               = id_key
        self.err_threshold        = err_threshold
        
        #get context and sequential features from config file 
        if not os.path.isfile(config_path):
            logging.error("The specified config path does not exist")
            raise FileNotFoundError("The specified config path does not exist")
        try:
            # Read the config file
            with open(config_path, 'r') as config_file:
                config = toml.load(config_file)

            # Get the context and sequential features
            self.context_features = config['context_features']
            self.sequential_features = config['sequential_features']

        except Exception as e:
            logging.error(f'Error while reading the config file: {str(e)}')
            raise e


        if metadata is not None:
            print('[INFO] {} samples loaded'.format(metadata.shape[0]))

        self.metadata['subset_0'] = ['full']*self.metadata.shape[0]

        os.makedirs(output_folder, exist_ok=True)
        
    @staticmethod
    def aux_serialize(sel : np.ndarray, path : str) -> None:
        if not isinstance(sel, np.ndarray):
            logging.error("Invalid data type provided to aux_serialize")
            raise ValueError("Invalid data type provided to aux_serialize")
        with tf.io.TFRecordWriter(path) as writer:
            for lc in sel:
                ex = DataPipeline.get_example(lc)
                writer.write(ex.SerializeToString())
        logging.info(f"wrote tfrecords to {path}")
         
        
    @staticmethod
    def get_example(self,row: pd.Series)-> tf.train.SequenceExample:
        """
        Converts a given row into a TensorFlow SequenceExample.

        Args:
            row (pd.Series): Row of data to be converted.

        Returns:
            tf.train.SequenceExample: The converted row as a SequenceExample.
        """
        dict_features = {}
        # Parse each context feature based on its dtype and add to the features dictionary
        for name in self.context_features:
            dict_features[name] = parse_dtype(row[name])

        # Create a context for the SequenceExample using the features dictionary
        element_context = tf.train.Features(feature=dict_features)

        dict_sequence = {}

        time = row[self.sequential_features[0]]
        mag = row[self.sequential_features[1]]

        lightcurve = [time, mag]


        # Create a sequence of features for each dimension of the lightcurve
        for col in range(len(lightcurve)):
            seqfeat = _float_feature(lightcurve[col][:])
            seqfeat = tf.train.FeatureList(feature=[seqfeat])
            dict_sequence['dim_{}'.format(col)] = seqfeat

        # Add the sequence to the SequenceExample
        element_lists = tf.train.FeatureLists(feature_list=dict_sequence)

        # Create the SequenceExample
        ex = tf.train.SequenceExample(context=element_context, feature_lists=element_lists)
        logging.info("Successfully converted to SequenceExample.")
        return ex
    
    
    def open_and_read_record(self, file_path : str) -> Any:
        """
        Opens and reads a .record file

        Args:
            file_path (str) : The path to the .record file to be read.

        Returns:
            raw_dataset (Any) : The raw dataset loaded from the .record file.
        """
        # Log the start of the file reading process
        logging.info(f'Starting to read the file from {file_path}.')

        # Use TensorFlow's TFRecordDataset method to read the .record file
        raw_dataset = tf.data.TFRecordDataset(file_path)

        # Log the completion of the file reading process
        logging.info(f'Successfully read the file from {file_path}.')

        return raw_dataset
    
    def train_val_test(self,
                       val_frac=0.2,
                       test_frac=0.2,
                       test_meta=None,
                       val_meta=None,
                       shuffle=True,
                       id_column_name=None,
                       k_fold=1):

        if id_column_name is None:
            id_column_name = self.metadata.columns[0]
        print('[INFO] Using {} col as sample identifier'.format(id_column_name))

        if (type(test_meta) is not list) and (k_fold > 1) and (type(test_meta) != type(None)):
            raise ValueError(f'k_fold={k_fold} does not match with number of test frames. Please provide a list of testing frames for each fold')
        if (type(val_meta) is not list) and (k_fold > 1) and (type(val_meta) != type(None)):
            raise ValueError(f'k_fold={k_fold} does not match with number of validation frames.Please, provide a list of validation frames for each fold')

        if test_meta is None: test_meta = []
        if val_meta is None: val_meta = []

        for k in range(k_fold):
            if shuffle:
                print('[INFO] Shuffling')
                self.metadata = self.metadata.sample(frac=1)

            try:
                test_meta[k]
            except:
                test_meta.append(self.metadata.sample(frac=test_frac))

            self.metadata = substract_frames(self.metadata, test_meta[k], on=id_column_name)

            try:
                val_meta[k]
            except:
                val_meta.append(self.metadata.sample(frac=val_frac))

            self.metadata = substract_frames(self.metadata, val_meta[k], on=id_column_name)

            self.metadata['subset_{}'.format(k)] = ['train']*self.metadata.shape[0]
            val_meta[k]['subset_{}'.format(k)]      = ['validation']*val_meta[k].shape[0]
            test_meta[k]['subset_{}'.format(k)]     = ['test']*test_meta[k].shape[0]

            self.metadata = pd.concat([self.metadata, val_meta[k], test_meta[k]])


    def prepare_data(self, container : np.ndarray, elements_per_shard : int, subset : str, fold_n : int):
        """Prepare the data to be saved as records"""
        if container is None or not hasattr(container, '__iter__'):
            raise ValueError("Invalid container provided to prepare_data")
        
        # Number of objects in the split
        N = self.metadata.shape[0]
        # Compute the number of shards

        n_shards = -np.floor_divide(N, -elements_per_shard)
        # Number of characters of the padded number of shards
        name_length = len(str(n_shards))

        # Create one file per shard
        shard_paths = []
        for shard in range(n_shards):
            # Get the shard number padded with 0s
            shard_name = str(shard+1).rjust(name_length, '0')
            # Get the shard store name
            shard_path= os.path.join(self.output_folder, subset+'_{}_{}.record'.format(fold_n, shard_name))
            # Save it into a list
            shard_paths.append(shard_path)

        shards_data = []
        for j in range(len(shard_paths)):
            sel = container[j*elements_per_shard:(j+1)*elements_per_shard]
            shards_data.append(sel)        
        return shards_data, shard_paths
    
    
    def read_all_parquets(self, path_parquets : str , metadata_path : str) -> pd.DataFrame:
        """
        Read the files from given paths and filters it based on err_threshold and ID from metadata
        Args:
            path_parquets (str): Directory path of parquet files
            metadata_path (str): File path of metadata
        Returns:
            new_df (pl.DataFrame): Processed dataframe
        """
        logging.info("Reading parquet files")

        if not os.path.exists(path_parquets):
            logging.error("The specified parquets path does not exist")
            raise FileNotFoundError("The specified parquets path does not exist")

        if not os.path.isfile(metadata_path):
            logging.error("The specified metadata path does not exist")
            raise FileNotFoundError("The specified metadata path does not exist")


        # Read the parquet filez lazily
        paths = os.path.join(path_parquets, '*.parquet')
        scan = pl.scan_parquet(paths)
        
        # Using partial information, extract only the necessary objects
        # Define filters
        ID_series = pl.Series(self.metadata[self.id_key].values)
        f1 = pl.col(self.id_key).is_in(ID_series)
        f2 = pl.col("err") < self.err_threshold  # Clean the data on the big lazy dataframe

        # Define groupby functions
        # func1 cleans the data
        func1 = lambda group_df: group_df.sort(self.sequential_features[0]) #mjd 

        # Filter, drop nulls, and sort every object
        new_df = scan.filter(f1 & f2).drop_nulls().groupby(self.id_key).apply(func1, schema=None)

        # Select only the relevant columns
        new_df = new_df.select([self.id_key] + self.sequential_features)

        # Mix metadata and the data
        new_df = new_df.groupby(self.id_key).all()
        # display(print(new_df))
        # First run takes more time!
        metadata_lazy = pl.scan_parquet(metadata_path, cache=True) # First run is slower
        # display(print(metadata_lazy))        
        # Perform the join to get the data
        new_df = new_df.join(other=metadata_lazy, on=self.id_key).collect(streaming=True) #streaming might be useless.                    
        return new_df
    


    def resample_folds(self, n_folds=1):
        print('[INFO] Creating {} random folds'.format(n_folds))
        print('Not implemented yet hehehe...')

    def run(self, path_parquets :str , metadata_path : str, n_jobs : int =1, elements_per_shard : int = 5000) -> None: 
        """
        Executes the DataPipeline operations which includes reading parquet files, processing samples and writing records.
        
        Args:
            path_parquets (str): Directory path of parquet files
            metadata_path (str): Path for metadata file
            n_jobs (int): The maximum number of concurrently running jobs. Default is 1
            elements_per_shard (int): Maximum number of elements per shard. Default is 5000
        """
        if not os.path.exists(path_parquets):
            logging.error("The specified parquets path does not exist")
            raise FileNotFoundError("The specified parquets path does not exist")
        
        if not os.path.isfile(metadata_path):
            logging.error("The specified metadata path does not exist")
            raise FileNotFoundError("The specified metadata path does not exist")
        
        # Start the operations
        logging.info("Starting DataPipeline operations")

        # threads = Parallel(n_jobs=n_jobs, backend='threading')
        fold_groups = [x for x in self.metadata.columns if 'subset' in x]
        pbar = tqdm(fold_groups, colour='#00ff00') # progress bar
        
        new_df = self.read_all_parquets(path_parquets, metadata_path)
        self.new_df = new_df

        for fold_n, fold_col in enumerate(pbar):
            pbar.set_description(f"Processing fold {fold_n+1}/{len(fold_groups)}")

            for subset in self.metadata[fold_col].unique():               
                # ============ Processing Samples ===========
                pbar.set_description("Processing {} {}".format(subset, fold_col))
                partial = self.metadata[self.metadata[fold_col] == subset]
                
                # Transform into a appropiate representation
                index = partial[self.id_key]
                b = np.isin(new_df[self.id_key].to_numpy(), index)
                container = new_df.filter(b).to_numpy()  
                
                # ============ Writing Records ===========

                pbar.set_description("Writting {} fold {}".format(subset, fold_n))
                output_file = os.path.join(self.output_folder, subset+'_{}.record'.format(fold_n))
                
                shards_data,shard_paths = self.prepare_data(container, elements_per_shard, subset, fold_n)
                    
                # Idea from
                with ThreadPoolExecutor(n_jobs) as exe:
                    # submit tasks to generate files
                    _ = [exe.submit(DataPipeline.aux_serialize, shard,shard_path) for shard, shard_path in zip(shards_data,shard_paths)]
    

        logging.info('Finished execution of DataPipeline operations')

def deserialize(sample, config_path = "./config.toml"):
    """
    Reads a serialized sample and converts it to tensor.
    Context and sequence features should match the name used when writing.
    Args:
        sample (binary): serialized sample

    Returns:
        type: decoded sample
    """
    # Read the configuration file
        # Load the configuration file
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
    except FileNotFoundError as e:
        logging.error(f'Configuration file not found at {config_path}. Please provide a valid path.')
        raise e
    except Exception as e:
        logging.error(f'An error occurred while loading the configuration file: {str(e)}')
        raise e

    # Define context features as strings
    context_features = {feat: tf.io.FixedLenFeature([], dtype=tf.string) for feat in config['context_features']}

    # Define sequence features as floating point numbers
    sequence_features = {feat: tf.io.VarLenFeature(dtype=tf.float32) for feat in config['sequential_features']}

    # Parse the serialized sample into context and sequence features
    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )
    
    # Cast context features to strings
    input_dict = {k: context[k] for k in config['context_features']}

    # Cast and store sequence features
    casted_inp_parameters = []
    for k in config['sequential_features']:
        print(f"Key: {k}")
        print(f"Sequence: {sequence[k]}")
        seq_dim = sequence[k]
        seq_dim = tf.sparse.to_dense(seq_dim)
        print(f"Dense sequence: {seq_dim}")
        seq_dim = tf.cast(seq_dim, tf.float32)
        print(f"Casted sequence: {seq_dim}")
        casted_inp_parameters.append(seq_dim)

    
    print(casted_inp_parameters)

    # Stack sequence features along a new third dimension
    sequence = tf.stack(casted_inp_parameters, axis=2)[0]
    # Add sequence to the input dictionary
    input_dict['input'] = sequence

    # Log the completion of deserialization
    logging.info(f'Successfully deserialized a sample.')
    return input_dict
