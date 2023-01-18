import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np

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
    """docstring for DataPipeline."""

    def __init__(self,
                 metadata=None,
                 context_features=None,
                 sequential_features=None,
                 output_folder='./records/output'):

        self.metadata             = metadata
        self.context_features     = context_features
        self.sequential_features  = sequential_features
        self.output_folder        = output_folder

        if metadata is not None:
            print('[INFO] {} samples loaded'.format(metadata.shape[0]))

        self.metadata['subset_0'] = ['full']*self.metadata.shape[0]

        os.makedirs(output_folder, exist_ok=True)

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

    def resample_folds(self, n_folds=1):
        print('[INFO] Creating {} random folds'.format(n_folds))
        print('Not implemented yet hehehe...')

    def run(self, n_jobs=1):
        threads = Parallel(n_jobs=n_jobs, backend='multiprocessing')
        fold_groups = [x for x in self.metadata.columns if 'subset' in x]

        pbar = tqdm(fold_groups, colour='#00ff00') # progress bar
        for fold_n, fold_col in enumerate(fold_groups):

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
                output_file = os.path.join(self.output_folder, subset+'_{}.record'.format(fold_n))
                with tf.io.TFRecordWriter(output_file) as writer:
                    for observations, context in var:
                        ex = DataPipeline.get_example(observations.values, context)
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
