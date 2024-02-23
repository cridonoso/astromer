import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import toml

import os

from joblib import wrap_non_picklable_objects
from joblib import Parallel, delayed
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm

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
        
        if metadata is not None:
            print('[INFO] {} samples loaded'.format(metadata.shape[0]))

        self.metadata['subset_0'] = ['full']*self.metadata.shape[0]

        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def get_example(lightcurve, metarow,
                    context_features, context_features_dtype, 
                    sequential_features, sequential_features_dtype):
        """
        Create a record example from numpy values.
        Serialization
        Args:
            lightcurve (DataFrame): time, magnitudes and observational error
            metarow (Series)
            context_features_values:
            context_features_dtype:
            sequential_features:
            sequential_features_dtype:
        Returns:
            tensorflow record example
        """
        dict_features = dict()
        for i, name in enumerate(context_features):
            dict_features[name] = parse_dtype(metarow[name], context_features_dtype[i])

        element_context = tf.train.Features(feature = dict_features)

        dict_sequence = dict()
        for i, name in enumerate(sequential_features):
            seqfeat = parse_dtype(lightcurve[name].tolist(), sequential_features_dtype[i])
            seqfeat = tf.train.FeatureList(feature = [seqfeat])
            dict_sequence[name] = seqfeat

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


    @staticmethod
    def lightcurve_step(row:pd.Series, context_features:list, sequential_features:list):
        observations = pd.read_csv(row['Path'])
        observations = observations.dropna()
        observations.sort_values('mjd')
        observations = observations.drop_duplicates(keep='last')
        observations = observations[sequential_features]
        return observations, row

    @classmethod
    def __process_sample__(cls, *args):
        var = cls.lightcurve_step(*args)
        return var

    def resample_folds(self, n_folds=1):
        print('[INFO] Creating {} random folds'.format(n_folds))
        print('Not implemented yet hehehe...')

    def run(self, n_jobs=1):
        threads = Parallel(n_jobs=n_jobs, backend='multiprocessing')
        fold_groups = [x for x in self.metadata.columns if 'subset' in x]

        pbar = tqdm(fold_groups, colour='#00ff00') # progress bar
        for fold_n, fold_col in enumerate(pbar):

            for subset in self.metadata[fold_col].unique():
                # ============ Processing Samples ===========
                pbar.set_description("Processing {} {}".format(subset, fold_col))
                partial = self.metadata[self.metadata[fold_col] == subset]

                var = threads(delayed(self.__process_sample__)(row,
                                                               self.context_features,
                                                               self.sequential_features)\
                                              for _, row in partial.iterrows())
                # ============ Writing Records ===========
                pbar.set_description("Writting {} fold {}".format(subset, fold_n))
                outfolder = os.path.join(self.output_folder, f'fold_{fold_n}', subset)
                os.makedirs(outfolder, exist_ok=True)
                shutil.copyfile(self.config_path, os.path.join(outfolder, 'config.toml'))
                with tf.io.TFRecordWriter(os.path.join(outfolder, 'out.record')) as writer:
                    for observations, metarow in var:
                        ex = DataPipeline.get_example(observations, 
                                                      metarow,
                                                      self.context_features, 
                                                      self.context_features_dtype,
                                                      self.sequential_features,
                                                      self.sequential_features_dtype)
                        writer.write(ex.SerializeToString())

        return var


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