import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import *

cwd = os.path.dirname(os.getcwd())
folder_data = cwd + "/generated_data/dataset_separated_by_ts/"


def create_ts_idx(idx, ts_ids_list, mask_dict=None):
    """
    idx_one = indices for one ts
    mask_dict = which samples to include for specific ts
    """
    
    s = []
    for ts_id in ts_ids_list:
        if mask_dict is None:
            s.append(pd.Series(ts_id, index=idx))
        else:
            s.append(pd.Series(ts_id, index=idx).iloc[mask_dict[ts_id]])
    s = pd.concat(s).rename("ts_id")
    return s.index, s


def G_block(input_size, output_size, hidden_units, block_layers, block_n):
    weight_decay = 1e-4
    
    def block(x):
        block_inputs = x[0]
        
        if block_n == 0: 
            x = Concatenate(name="concat")(x)
        else:
            x = x[0]        
        
        for FC_n in range(1, block_layers+1):
            x = Dense(hidden_units, "relu", 
                      name="B{}_FC{}".format(block_n, FC_n),
                      kernel_regularizer=l1(weight_decay),
                     )(x)

        x_hat = Dense(input_size[0], "linear", 
                      name="B{}_gb".format(block_n))(x)

        y_hat = Dense(output_size, "linear", 
                      name="B{}_gf".format(block_n))(x)
        
        x_hat = Subtract(name="subtract_B{}".format(block_n))([block_inputs, x_hat])
        x_hat = tf.keras.layers.Activation('relu')(x_hat)
        return [x_hat, y_hat]
    
    inputs_lags = Input((input_size[0],), name="B{}_lags".format(block_n))
    inputs_exog = Input((input_size[1],), name="B{}_exog".format(block_n))
    inputs = [inputs_lags, inputs_exog]
        
    outputs = block(inputs)
    model = Model(inputs, outputs, name="B{}".format(block_n))
    return model

def G_stack(input_size, output_size, block_layers, hidden_units, n_blocks, block_sharing):
    def stack(x):

        x_lags = x[0]
        x_exog = x[1]

        # preprocess only x_lags
        level = tf.reduce_max(x_lags, axis=-1, keepdims=True)
        x_lags= tf.math.divide_no_nan(x_lags, level)  # normalize
        x = [x_lags, x_exog]

        block_list = [G_block(input_size, output_size, hidden_units, block_layers, block_n=i) for i in range(n_blocks)]

        y_hat_list = []
        for block_n in range(n_blocks):
            if block_n == 0:
                x, y_hat = block_list[block_n](x)
            else:
                x, y_hat = block_list[block_n]([x, x_exog])

            y_hat_list.append(y_hat)

        y_hat_sum = Add(name="add")(y_hat_list)
        return y_hat_sum * level
    return stack

def NBeats_exog(params):
    inputs_lags = Input((params["input_size"][0],), name="lags")
    inputs_exog = Input((params["input_size"][1],), name="exog")
    inputs = [inputs_lags, inputs_exog]

    outputs = G_stack(params["input_size"], 
                      params["output_size"], 
                      params["block_layers"], 
                      params["hidden_units"], 
                      params["n_blocks"], 
                      None)(inputs)

    model = Model(inputs, outputs)
    return model


class TSGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras. Train & trainval sampling is stochastic, val / test is not!'
    def __init__(self, batch_size=1024, n_steps_per_epoch=50, 
                 ts_ids_list=[], set_type="train", predict=False):
                 
        self.batch_size = batch_size
        self.ts_ids_list = ts_ids_list  # number of uniques ts in a set
        self.set_type = set_type
        self.predict = predict
        
        # only one pass over dataset during val and test or predict
        if (self.set_type in ["val", "test"]) or self.predict:
            self.n_steps_per_epoch = len(ts_ids_list)
        else: self.n_steps_per_epoch = n_steps_per_epoch
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_steps_per_epoch
            
    def get_ts_batch(self, ts_id, n_series, return_all_rows=False):
        """
        Load n_series of rows from train dataset for desired ts_id.
        
        """
        
        X_lags = np.load(folder_data + "X_{}_ts_id={}, lags.npy".format(self.set_type, ts_id))
        X_exog = np.load(folder_data + "X_{}_ts_id={}, exog.npy".format(self.set_type, ts_id))
        y_ts = np.load(folder_data + "y_{}_ts_id={}.npy".format(self.set_type, ts_id))

        chosen_rows = np.random.choice(np.arange(len(y_ts)), n_series)
        
        # whether to return randomly sampled rows arr all data for each series
        if (self.set_type in ["val", "test"]) or self.predict or return_all_rows:
            return [X_lags, X_exog], y_ts
        else:
            return [X_lags[chosen_rows], X_exog[chosen_rows]], y_ts[chosen_rows]
              
    
    def __data_generation(self, index):
        """
        Loads desired data from a disk and returns a mini-batch.
        If val, test or predict -> one batch holds data for one ts, otherwise
        one batch holds sampled data from multiple ts.
        """

        if (self.set_type in ["val", "test"]) or self.predict:
            [X_batch_lags, X_batch_exog], y_batch = self.get_ts_batch(ts_id=self.ts_ids_list[index], n_series=1)
            
        else:
            sampling_info = (pd.Series(np.random.choice(self.ts_ids_list,
                                       size=self.batch_size))
                               .value_counts().sort_index())
            [X_batch_lags, X_batch_exog], y_batch = [[], []], []

            for ts_id in sampling_info.index:
                n_series = sampling_info.loc[ts_id]  #  number of timestamps to sample

                [X_ts_lags, X_ts_exog], y_ts = self.get_ts_batch(ts_id, n_series)
                X_batch_lags.append(X_ts_lags)
                X_batch_exog.append(X_ts_exog)
                
                y_batch.append(y_ts)

            X_batch_lags = np.concatenate(X_batch_lags)
            X_batch_exog = np.concatenate(X_batch_exog)
            y_batch = np.concatenate(y_batch)

        return [X_batch_lags, X_batch_exog], y_batch

    
    def __getitem__(self, index):
        """
        index: consecutive batch number (0, 1, 2, ...) in one epoch.
        This method is executed at the beginning of every batch &
        returns a mini-batch of data.
        """
        # Generate data
        [X_lags, X_exog], y = self.__data_generation(index)
        return [X_lags, X_exog], y