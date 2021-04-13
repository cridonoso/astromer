import tensorflow as tf 
import pandas as pd
import os

from core.masking import create_padding_mask


def plot_lc(batch, ax, target, n=0, title=''):
    mask_1 = create_padding_mask(batch['serie_1'], batch['steps_1'])
    mask_2 = create_padding_mask(batch['serie_2'], batch['steps_2'])

    serie_1 = tf.boolean_mask(batch['serie_1'][n], tf.logical_not(mask_1[n]))
    serie_2 = tf.boolean_mask(batch['serie_2'][n], tf.logical_not(mask_2[n]))
    
    y = batch['label'][n]
    objects = pd.read_csv(os.path.join(target, 'objects.csv'))
    cls_label = objects.iloc[int(y)]['label']
       
    ax.plot(serie_1[:, 0], serie_1[:, 1])
    ax.plot(serie_2[:, 0], serie_2[:, 1])
    ax.set_title(cls_label+' '+title)
    return ax

def plot_input_layer(in_dict, ax, input_len, n):
    serie_1 = in_dict['inputs'][n][1:input_len//2+1]
    times_1 = in_dict['times'][n][1:input_len//2+1]
    mask_1 = in_dict['tar_mask'][n][1:input_len//2+1]
    s1 = tf.concat([times_1, serie_1], 1)
    s1 = tf.boolean_mask(s1, mask_1)
    
    serie_2 = in_dict['inputs'][n][input_len//2+2:]
    times_2 = in_dict['times'][n][input_len//2+2:]
    mask_2 = in_dict['tar_mask'][n][input_len//2+2:]
    

    s2 = tf.concat([times_2, serie_2], 1)
    s2 = tf.boolean_mask(s2, mask_2)

    ax.plot(s1[:, 0], s1[:, 1], '.--')
    ax.plot(s2[:, 0], s2[:, 1], '.--')
    
    title = 'NonRandom Next Portion'
    if int(in_dict['target'][n][0,0]) == 1:
        title = 'Random Next Portion'
    ax.set_title(title)
    return ax