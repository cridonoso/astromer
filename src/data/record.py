import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import polars as pl
import numpy as np
import logging
import shutil
import random
import glob
import toml
import os

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from joblib import Parallel, delayed
from typing import List, Dict, Any
from io import BytesIO
from tqdm import tqdm

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

def parse_dtype(value, data_type):
    if type(value) in [int, float] and data_type == 'integer':
        return _int64_feature([int(value)])
    if type(value) in [int, float] and data_type == 'float':
        return _float_feature([value])
    if type(value) == str and data_type == 'string':
        return _bytes_feature([str(value).encode()])

    if type(value) == list:
        if type(value[0]) == int and data_type == 'integer':
            return _int64_feature(value)
        if type(value[0]) == float and data_type == 'float':
            return _float_feature(value)
        if type(value[0]) == str and data_type == 'string':
            return _bytes_feature(value)

    raise ValueError('[ERROR] {} with type {} could not be parsed. Please use <str>, <int>, or <float>'.format(value, type(value)))

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
        metadata
        config_path
    """

    def __init__(self,
                 metadata=None,
                 config_path= "./config.toml"):

        

        #get context and sequential features from config file 
        if not os.path.isfile(config_path):
            logging.error("The specified config path does not exist")
            raise FileNotFoundError("The specified config path does not exist")

        # Read the config file
        with open(config_path, 'r') as f:
            config = toml.load(f)

        # Saving class variables
        self.metadata                  = metadata
        self.config_path               = config_path
        self.config                    = config
        self.context_features          = config['context_features']['value']
        self.context_features_dtype    = config['context_features']['dtypes']
        self.sequential_features       = config['sequential_features']['value']
        self.sequential_features_dtype = config['sequential_features']['dtypes']
        self.output_folder             = config['general']['target']['value']
        self.id_column                 = config['general']['id_column']['value']

        assert self.metadata[self.id_column].dtype == int, \
        'ID column should be an integer Serie but {} was given'.format(self.metadata[self.id_column].dtype)
        
        if metadata is not None:
            print('[INFO] {} samples loaded'.format(metadata.shape[0]))

        self.metadata['subset_0'] = ['full']*self.metadata.shape[0]

        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def aux_serialize(sel : pl.DataFrame, 
                      path : str, 
                      context_features: list, 
                      context_features_dtype: list,
                      sequential_features: list,
                      sequential_features_dtype: list) -> None:
        if not isinstance(sel, pl.DataFrame):
            logging.error("Invalid data type provided to aux_serialize")
            raise ValueError("Invalid data type provided to aux_serialize")

        with tf.io.TFRecordWriter(path) as writer:
            for row  in sel.iter_rows(named=True):
                ex = DataPipeline.get_example(row, context_features, context_features_dtype, 
                                              sequential_features, sequential_features_dtype)
                writer.write(ex.SerializeToString())
         
        
    @staticmethod
    def get_example(row: dict, 
                    context_features: list, 
                    context_features_dtype: list,
                    sequential_features: list,
                    sequential_features_dtype: list) -> tf.train.SequenceExample:
        """
        Converts a given row into a TensorFlow SequenceExample.

        Args:
            row (pd.Series): Row of data to be converted.

        Returns:
            tf.train.SequenceExample: The converted row as a SequenceExample.
        """
        dict_features = {}
        # Parse each context feature based on its dtype and add to the features dictionary
        for name, data_type in zip(context_features, context_features_dtype):
            dict_features[name] = parse_dtype(row[name], data_type=data_type)

        # Create a context for the SequenceExample using the features dictionary
        element_context = tf.train.Features(feature=dict_features)

        dict_sequence = {}
        # Create a sequence of features for each dimension of the lightcurve
        for col, data_type in zip(sequential_features, sequential_features_dtype):
            seqfeat = parse_dtype(row[col][:], data_type=data_type)
            seqfeat = tf.train.FeatureList(feature=[seqfeat])
            dict_sequence[col] = seqfeat

        # Add the sequence to the SequenceExample
        element_lists = tf.train.FeatureLists(feature_list=dict_sequence)

        # Create the SequenceExample
        ex = tf.train.SequenceExample(context=element_context, feature_lists=element_lists)
        # logging.info("Successfully converted to SequenceExample.")
        return ex
    
    def inspect_records(self, dir_path:str = './records/output/', num_records: int = 1):
        """
        Function to inspect the first 'num_records' from a random TFRecord file in the given directory.

        Args:
            dir_path (str): Directory path where TFRecord files are located.
            num_records (int): Number of records to inspect.

        Returns:
            NoReturn
        """
        # Use glob to get all the .record files in the directory
        file_paths = glob.glob(dir_path + '*.record')

        # Select a random file path
        file_path = random.choice(file_paths)

        try:
            raw_dataset = tf.data.TFRecordDataset(file_path)
            for raw_record in raw_dataset.take(num_records):
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())

            logging.info(f'Successfully inspected {num_records} records from {file_path}.')
        except Exception as e:
            logging.error(f'Error while inspecting records. Error message: {str(e)}')
            raise e


    
    def train_val_test(self,
                       val_frac=0.2,
                       test_frac=0.2,
                       test_meta=None,
                       val_meta=None,
                       shuffle=True,
                       id_column_name=None,
                       k_fold=1):

        if k_fold > 1 and test_meta is not None:
            assert len(test_meta) == k_fold, 'test_meta should be a list containing {}-fold subsets'.format(k_fold)

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

            self.metadata['subset_{}'.format(k)]    = ['train']*self.metadata.shape[0]
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
        print(self.output_folder, 'fold_'+str(fold_n), subset)
        root = os.path.join(self.output_folder, 'fold_'+str(fold_n), subset)
        os.makedirs(root, exist_ok=True)
        shutil.copyfile(self.config_path, os.path.join(root, 'config.toml'))
        for shard in range(n_shards):
            # Get the shard number padded with 0s
            shard_name = str(shard+1).rjust(name_length, '0')
            # Get the shard store name
            shard_path= os.path.join(root, '{}.record'.format(shard_name))
            # Save it into a list
            shard_paths.append(shard_path)

        shards_data = []
        for j in range(len(shard_paths)):
            sel = container[j*elements_per_shard:(j+1)*elements_per_shard]
            shards_data.append(sel)        
        return shards_data, shard_paths
    
    def lightcurve_step(self, inputs):
        """
        Preprocessing applied to each light curve separately
        """
        # First feature is time
        inputs = inputs.sort(self.sequential_features[0]) 
        return inputs

    def observations_step(self):
        """
        Preprocessing applied to all observations. Filter only
        """
        fn = pl.col("err") < 1.  # Clean the data on the big lazy dataframe
        return fn

    def read_all_parquets(self, observations_path : str) -> pd.DataFrame:
        """
        Read the files from given paths and filters it based on err_threshold and ID from metadata
        Args:
            observations_path (str): Directory path of parquet files
        Returns:
            new_df (pl.DataFrame): Processed dataframe
        """
        # logging.info("Reading parquet files")

        if not os.path.exists(observations_path):
            logging.error("The specified parquets path does not exist")
            raise FileNotFoundError("The specified parquets path does not exist")


        # Read the parquet filez lazily
        paths = os.path.join(observations_path, '*.parquet')
        scan = pl.scan_parquet(paths)

        # Using partial information, extract only the necessary objects
        ID_series = pl.Series(self.metadata[self.id_column].values)
        f1 = pl.col(self.id_column).is_in(ID_series)
        scan.filter(f1)
        
        lightcurves_fn  = lambda light_curve: self.lightcurve_step(light_curve)

        # Filter, drop nulls, and sort every object
        processed_obs = scan.filter(self.observations_step()).drop_nulls().groupby(self.id_column).apply(lightcurves_fn, schema=None)

        # Select only the relevant columns
        processed_obs = processed_obs.select([self.id_column] + self.sequential_features)

        # Mix metadata and the data
        processed_obs = processed_obs.groupby(self.id_column).all()

        # First run takes more time!
        # metadata_lazy = pl.scan_parquet(metadata_path, cache=True) # First run is slower
        metadata_lazy = pl.from_pandas(self.metadata).lazy()   
        # Perform the join to get the data
        processed_obs = processed_obs.join(other=metadata_lazy, 
                                            on=self.id_column).collect(streaming=False) #streaming might be useless.                    
        
        return processed_obs
    


    def resample_folds(self, n_folds=1):
        print('[INFO] Creating {} random folds'.format(n_folds))
        print('Not implemented yet hehehe...')

    def run(self, observations_path :str , metadata_path : str, n_jobs : int =1, elements_per_shard : int = 5000) -> None: 
        """
        Executes the DataPipeline operations which includes reading parquet files, processing samples and writing records.
        
        Args:
            observations_path (str): Directory path of parquet files containing light curves observations
            metadata_path (str): Path for metadata file
            n_jobs (int): The maximum number of concurrently running jobs. Default is 1
            elements_per_shard (int): Maximum number of elements per shard. Default is 5000
        """
        if not os.path.exists(observations_path):
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
        
        print('[INFO] Reading parquet')
        new_df = self.read_all_parquets(observations_path)
        print('[INFO] Light curves loaded')
        self.new_df = new_df

        for fold_n, fold_col in enumerate(pbar):
            pbar.set_description(f"Processing fold {fold_n}/{len(fold_groups)}")
            for subset in self.metadata[fold_col].unique():               
                # ============ Processing Samples ===========
                partial = self.metadata[self.metadata[fold_col] == subset]
                
                # Transform into a appropiate representation
                index = partial[self.id_column]
                b = np.isin(new_df[self.id_column].to_numpy(), index)
                container = new_df.filter(b)
                
                # ============ Writing Records ===========                
                shards_data, shard_paths = self.prepare_data(container, elements_per_shard, subset, fold_n)
                
                # for shard, shard_path in zip(shards_data,shard_paths):
                #     DataPipeline.aux_serialize(shard, shard_path, 
                #                     self.context_features, self.context_features_dtype, 
                #                     self.sequential_features, self.sequential_features_dtype)

                with ThreadPoolExecutor(n_jobs) as exe:
                    # submit tasks to generate files
                    _ = [exe.submit(DataPipeline.aux_serialize, shard, shard_path, 
                                    self.context_features, self.context_features_dtype, 
                                    self.sequential_features, self.sequential_features_dtype) \
                             for shard, shard_path in zip(shards_data,shard_paths)]
    

        logging.info('Finished execution of DataPipeline operations')




def get_tf_dtype(data_type, is_sequence=False):
    if not is_sequence:
        if data_type == 'integer': return tf.io.FixedLenFeature([], dtype=tf.int64)
        if data_type == 'float': return tf.io.FixedLenFeature([], dtype=tf.float32)
        if data_type == 'string': return tf.io.FixedLenFeature([], dtype=tf.string)
    else:
        if data_type == 'integer': return tf.io.VarLenFeature(dtype=tf.int64)
        if data_type == 'float': return tf.io.VarLenFeature(dtype=tf.float32)
        if data_type == 'string': return tf.io.VarLenFeature(dtype=tf.string32)

def deserialize(sample, records_path=None):
    """
    Reads a serialized sample and converts it to tensor.
    Context and sequence features should match the name used when writing.
    Args:
        sample (binary): serialized sample

    Returns:
        type: decoded sample
    """
    try:
        with open(os.path.join(records_path, 'config.toml'), 'r') as f:
            config = toml.load(f)
            print(config)
    except FileNotFoundError as e:
        logging.error(f'Configuration file not found at {records_path}. Please provide a valid path.')
        raise e
    except Exception as e:
        logging.error(f'An error occurred while loading the configuration file: {str(e)}')
        raise e

    # Define context features as strings
    context_features = {}
    for feat, data_type in zip(config['context_features']['value'], config['context_features']['dtypes']):
        context_features[feat]= get_tf_dtype(data_type)

    sequence_features = {}
    # Define sequence features as floating point numbers
    for feat, data_type in zip(config['sequential_features']['value'], config['sequential_features']['dtypes']):
        sequence_features[feat]= get_tf_dtype(data_type, is_sequence=True)

    # Parse the serialized sample into context and sequence features
    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )
    
    # Cast context features to strings
    input_dict = {k: context[k] for k in config['context_features']['value']}

    # Cast and store sequence features
    casted_inp_parameters = []
    for k in config['sequential_features']['value']:
        seq_dim = sequence[k]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)

    # Add sequence to the input dictionary
    input_dict['input'] = tf.stack(casted_inp_parameters, axis=2)
    return input_dict