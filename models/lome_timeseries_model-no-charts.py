#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
print(tf.__version__)


import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Sequential

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell
from sklearn.preprocessing import MinMaxScaler


import warnings
warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (10, 7.5)
plt.rcParams['axes.grid'] = False

tf.random.set_seed(42)
np.random.seed(42)




train_df = pd.read_csv('data/lome_train1.csv', index_col='date_time',parse_dates=True)
val_df = pd.read_csv('data/lome_val1.csv', index_col='date_time',parse_dates=True)
test_df = pd.read_csv('data/lome_test1.csv', index_col='date_time',parse_dates=True)

print(train_df.shape, val_df.shape, test_df.shape)


#DataWindow Class


class DataWindow():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='rain_sum (mm)', max_subplots=3):
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
              label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
              label_col_index = plot_col_index

            if label_col_index is None:
              continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s', label='Labels', c='green', s=64)
            if model is not None:
              predictions = model(inputs)
              plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='red', s=64)

            if n == 0:
              plt.legend()

        plt.xlabel('Time (h)')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result





class Baseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs

        elif isinstance(self.label_index, list):
            tensors = []
            for index in self.label_index:
                result = inputs[:, :, index]
                result = result[:, :, tf.newaxis]
                tensors.append(result)
            return tf.concat(tensors, axis=-1)

        result = inputs[:, :, self.label_index]
        return result[:,:,tf.newaxis]


# Multi-output baseline model



mo_single_step_window = DataWindow(input_width=1, label_width=1, shift=1, label_columns=['precipitation_sum (mm)','rain_sum (mm)','river_discharge','intensity_rain','intensity_flood','intensity_drought'])
mo_wide_window = DataWindow(input_width=14, label_width=14, shift=1, label_columns=['precipitation_sum (mm)','rain_sum (mm)','river_discharge','intensity_rain','intensity_flood','intensity_drought'])



column_indices = {name: i for i, name in enumerate(train_df.columns)}



print(column_indices['precipitation_sum (mm)'])
print(column_indices['rain_sum (mm)'])
print(column_indices['river_discharge'])
print(column_indices['intensity_rain'])
print(column_indices['intensity_flood'])
print(column_indices['intensity_drought'])




mo_baseline_last = Baseline(label_index=[7,8,17,18,19,20])

mo_baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])



def compile_and_fit(model, window, patience=3, max_epochs=50):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train,
                       epochs=max_epochs,
                       validation_data=window.val,
                       callbacks=[early_stopping])
    

    return history



mo_dense = Sequential([
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=6,activation='relu')
])

history = compile_and_fit(mo_dense, mo_single_step_window)



# **LSTM**




mo_lstm_model = Sequential([
    LSTM(32, return_sequences=True),
    Dense(units = 6,activation='relu')
])

history = compile_and_fit(mo_lstm_model, mo_wide_window)



custom_mo_wide_window = DataWindow(input_width=14, label_width=14, shift=14, label_columns=['precipitation_sum (mm)','rain_sum (mm)','river_discharge','intensity_rain','intensity_flood','intensity_drought'])
input_indices = custom_mo_wide_window.input_indices
label_indices = custom_mo_wide_window.label_indices


predicted_results = mo_lstm_model.predict(custom_mo_wide_window.test)
predicted_array= predicted_results[0]

my_array = np.array(predicted_array)

df = pd.DataFrame(my_array)

df_columns =df.rename(columns={0: "precipitation_sum (mm)", 1: "rain_sum (mm)",2:"river_discharge",3:"intensity_rain",4:"intensity_flood",5:"intensity_drought"})

df_columns.head(14)

