import zipfile
from io import BytesIO
import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

from joblib import wrap_non_picklable_objects
from joblib import Parallel, delayed
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
from time import time


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

class DataPipeline:
    """docstring for DataPipeline."""

    def __init__(self,
                 metadata=None,
                 context_features=None,
                 sequential_features=None,
                 output_file='output.records'):

        self.metadata             = metadata
        self.context_features     = context_features
        self.sequential_features  = sequential_features
        self.output_file          = output_file

        if metadata is not None:
            print('[INFO] {} samples loaded'.format(metadata.shape[0]))

    @staticmethod
    def get_example(lightcurve, context_features_values):
        """
        Create a record example from numpy values.
        Serialization
        Args:
            lightcurve (numpy array): time, magnitudes and observational error
            context_features_values: NONE
        Returns:
            tensorflow record example
        """
        dict_features = dict()
        for name, value in context_features_values.items():
            dict_features[name] = parse_dtype(value)

        element_context = tf.train.Features(feature = dict_features)

        dict_sequence = dict()
        for col in range(lightcurve.shape[1]):
            seqfeat = _float_feature(lightcurve[:, col])
            seqfeat = tf.train.FeatureList(feature = [seqfeat])
            dict_sequence['dim_{}'.format(col)] = seqfeat

        element_lists = tf.train.FeatureLists(feature_list=dict_sequence)
        ex = tf.train.SequenceExample(context = element_context,
                                      feature_lists= element_lists)
        return ex

    @staticmethod
    def process_sample(row:pd.Series, context_features:list, sequential_features:list):
        observations = pd.read_csv(row['Path'])
        observations.columns = ['mjd', 'mag', 'errmag']
        observations = observations.dropna()
        observations.sort_values('mjd')
        observations = observations.drop_duplicates(keep='last')
        observations = observations[sequential_features]
        context_features_values = row[context_features].to_dict()
        return observations, context_features_values

    @classmethod
    def __process_sample__(cls, *args):
        var = cls.process_sample(*args)
        return var

    def run(self, n_jobs=1):
        print('[INFO] Processing data...')
        threads = Parallel(n_jobs=n_jobs, backend='multiprocessing')
        var = threads(delayed(self.__process_sample__)(row,
                                                       self.context_features,
                                                       self.sequential_features)\
                                      for _, row in self.metadata.iterrows())

        print('[INFO] Writing records...')
        with tf.io.TFRecordWriter(self.output_file) as writer:
            for observations, context in tqdm(var):
                ex = DataPipeline.get_example(observations.values, context)
                writer.write(ex.SerializeToString())

        return var


def get_example(lightcurve, context_features_values):
    """
    Create a record example from numpy values.
    Serialization
    Args:
        lightcurve (numpy array): time, magnitudes and observational error
        context_features_values: NONE
    Returns:
        tensorflow record example
    """
    dict_features = dict()
    for name, value in context_features_values.items():
        dict_features[name] = parse_dtype(value)

    element_context = tf.train.Features(feature = dict_features)

    dict_sequence = dict()
    for col in range(lightcurve.shape[1]):
        seqfeat = _float_feature(lightcurve[:, col])
        seqfeat = tf.train.FeatureList(feature = [seqfeat])
        dict_sequence['dim_{}'.format(col)] = seqfeat

    element_lists = tf.train.FeatureLists(feature_list=dict_sequence)
    ex = tf.train.SequenceExample(context = element_context,
                                  feature_lists= element_lists)
    return ex

def write_records(frame, dest, max_lcs_per_record, source, unique, zip_flag=False, n_jobs=None, max_obs=200, **kwargs):
    # Get frames with fixed number of lightcurves
    collection = [frame.iloc[i:i+max_lcs_per_record] \
                  for i in range(0, frame.shape[0], max_lcs_per_record)]
    
    if zip_flag == True:
        with zipfile.ZipFile(source.split('/')[0]+'.zip', 'r') as zp:

            for counter, subframe in enumerate(collection):
                var = [process_lc2(row, source, unique, zip_flag, file_ins = zp, **kwargs) for k, row in subframe.iterrows()]

                with tf.io.TFRecordWriter(dest+'/chunk_{}.record'.format(counter)) as writer:
                    for counter2, data_lc in enumerate(var):
                        process_lc3(*data_lc, writer)    
    
    else:

        for counter, subframe in enumerate(collection):
            #-------#
            var = Parallel(n_jobs=n_jobs)(delayed(process_lc2)(row, source, unique, zip_flag, file_ins = None, **kwargs) \
                                         for k, row in subframe.iterrows())
            #------#
            
            #var = Parallel(backend = 'threading', n_jobs=n_jobs)(delayed(process_lc2)(row, source, unique, zp, **kwargs) \
             #                      for k, row in subframe.iterrows())
            

            with tf.io.TFRecordWriter(dest+'/chunk_{}.record'.format(counter)) as writer:
                for counter2, data_lc in enumerate(var):
                    process_lc3(*data_lc, writer)

def create_dataset(meta_df,
                   source,
                   target='records/new_ztf_g',
                   n_jobs=None,
                   subsets_frac=(0.5, 0.25),
                   test_subset=None,
                   max_lcs_per_record=100,
                   **kwargs): # kwargs contains additional arguments for the read_csv() function

    os.makedirs(target, exist_ok=True)
    zip_flag = False
    if source.split('.')[-1] == 'zip':
        zip_flag = True
        source = source.split('.')[0]+'/'    

    bands = meta_df['Band'].unique()
    if len(bands) > 1:
        b = input('Filters {} were found. Type one to continue'.format(' and'.join(bands)))
        meta_df = meta_df[meta_df['Band'] == b]

    unique, counts = np.unique(meta_df['Class'], return_counts=True)
    info_df = pd.DataFrame()
    info_df['label'] = unique
    info_df['size'] = counts
    info_df.to_csv(os.path.join(target, 'objects.csv'), index=False)

    # Separate by class
    cls_groups = meta_df.groupby('Class')

    for cls_name, cls_meta in tqdm(cls_groups, total=len(cls_groups)):
        subsets = divide_training_subset(cls_meta,
                                         train=subsets_frac[0],
                                         val=subsets_frac[1],
                                         test_meta = test_subset)

        for subset_name, frame in subsets:
            dest = os.path.join(target, subset_name, cls_name)
            if not os.path.exists(dest):
                os.makedirs(dest, exist_ok=True)
            write_records(frame, dest, max_lcs_per_record, source, unique, zip_flag, n_jobs, **kwargs)


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
