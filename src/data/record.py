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
                 context_features=None,
                 sequential_features=None,
                 output_folder='./records/output', 
                 id_key = 'newID', 
                 err_threshold=1, ):

        self.metadata             = metadata
        self.context_features     = context_features
        self.sequential_features  = sequential_features
        self.output_folder        = output_folder
        self.id_key               = id_key
        self.err_threshold = err_threshold

        if metadata is not None:
            print('[INFO] {} samples loaded'.format(metadata.shape[0]))

        self.metadata['subset_0'] = ['full']*self.metadata.shape[0]

        os.makedirs(output_folder, exist_ok=True)
        
    @staticmethod
    def aux_serialize(sel, path):
        with tf.io.TFRecordWriter(path) as writer:
            for lc in sel:
                ex = DataPipeline.get_example(lc)
                writer.write(ex.SerializeToString())       
        
    @staticmethod
    def get_example(self, row, context_features=['ID', 'Label', 'Class']):
        
        """
        Create a record example from numpy values in a list of dictionaries.
        Serialization
        Args:
            row (dictionary): Keys ['ID', 'mjd', 'mag', 'Class', 'Path', 'Band', 'newID', 'Label'])
            context_features_values: ['ID', 'Label', 'Class']
        Returns:
            tensorflow record example
        """
        dict_features = dict()
        for name in context_features:
            dict_features[name] = parse_dtype(row[name])

        element_context = tf.train.Features(feature = dict_features)

        dict_sequence = dict()
        time = row[self.sequential_features[0]]
        mag = row[self.sequential_features[1]]
        lightcurve = [time, mag]
        for col in range(len(lightcurve)):
            seqfeat = _float_feature(lightcurve[col][:])
            seqfeat = tf.train.FeatureList(feature = [seqfeat])
            dict_sequence['dim_{}'.format(col)] = seqfeat

        element_lists = tf.train.FeatureLists(feature_list=dict_sequence)
        ex = tf.train.SequenceExample(context = element_context,
                                      feature_lists= element_lists)
        return ex    

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


    def prepare_data(self, container, elements_per_shard, subset, fold_n):
        """Prepare the data to be saved as records"""
        
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
    
    
    def read_all_parquets(self, path_parquets, metadata_path):
        """
        Read the files from given paths and filters it based on err_threshold and ID from metadata
        Args:
            path_parquets (str): Directory path of parquet files
            metadata_path (str): File path of metadata
        Returns:
            new_df (pl.DataFrame): Processed dataframe
        """
        logging.info("Reading parquet files")

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

    def run(self, path_parquets, metadata_path, n_jobs=1, elements_per_shard=5000):
        # threads = Parallel(n_jobs=n_jobs, backend='threading')
        fold_groups = [x for x in self.metadata.columns if 'subset' in x]
        

        pbar = tqdm(fold_groups, colour='#00ff00') # progress bar
        
        new_df = self.read_all_parquets(path_parquets, metadata_path)
        self.new_df = new_df
        for fold_n, fold_col in enumerate(fold_groups):

            for subset in self.metadata[fold_col].unique():               
                # ============ Processing Samples ===========
                pbar.set_description("Processing {} {}".format(subset, fold_col))
                partial = self.metadata[self.metadata[fold_col] == subset]
                
                # Transform into a appropiate representation
                index = partial.newID
                b = np.isin(new_df['newID'].to_numpy(), index)
                container = new_df.filter(b).to_numpy()  
                
                # ============ Writing Records ===========
                pbar.set_description("Writting {} fold {}".format(subset, fold_n))
                output_file = os.path.join(self.output_folder, subset+'_{}.record'.format(fold_n))
                
                shards_data,shard_paths = self.prepare_data(container, elements_per_shard, subset, fold_n)
                    
                # Idea from
                with ThreadPoolExecutor(n_jobs) as exe:
                    # submit tasks to generate files
                    _ = [exe.submit(DataPipeline.aux_serialize, shard,shard_path) for shard, shard_path in zip(shards_data,shard_paths)]
                

def deserialize(sample):
    """
    Read a serialized sample and convert it to tensor
    Context and sequence features should match with the name used when writing.
    Args:
        sample (binary): serialized sample

    Returns:
        type: decoded sample
    """
    context_features = {'label': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'length': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = dict()
    for i in range(3):
        sequence_features['dim_{}'.format(i)] = tf.io.VarLenFeature(dtype=tf.float32)

    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )

    input_dict = dict()
    input_dict['lcid']   = tf.cast(context['id'], tf.string)
    input_dict['length'] = tf.cast(context['length'], tf.int32)
    input_dict['label']  = tf.cast(context['label'], tf.int32)

    casted_inp_parameters = []
    for i in range(3):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)


    sequence = tf.stack(casted_inp_parameters, axis=2)[0]
    input_dict['input'] = sequence
    return input_dict
