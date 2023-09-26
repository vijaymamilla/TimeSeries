#!/usr/bin/env python
# coding: utf-8

# In[59]:




import tensorflow as tf
print(tf.__version__)


# In[60]:


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


# In[ ]:





# In[61]:


plt.rcParams['figure.figsize'] = (10, 7.5)
plt.rcParams['axes.grid'] = False


# In[62]:


tf.random.set_seed(42)
np.random.seed(42)


# In[63]:


train_df = pd.read_csv('data/lome_train1.csv', index_col='date_time',parse_dates=True)
val_df = pd.read_csv('data/lome_val1.csv', index_col='date_time',parse_dates=True)
test_df = pd.read_csv('data/lome_test1.csv', index_col='date_time',parse_dates=True)

print(train_df.shape, val_df.shape, test_df.shape)


# In[64]:


#DataWindow Class


# 

# In[65]:


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


# In[66]:


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
# 

# In[67]:


mo_single_step_window = DataWindow(input_width=1, label_width=1, shift=1, label_columns=['precipitation_sum (mm)','rain_sum (mm)','river_discharge','intensity_rain','intensity_flood','intensity_drought'])
mo_wide_window = DataWindow(input_width=14, label_width=14, shift=1, label_columns=['precipitation_sum (mm)','rain_sum (mm)','river_discharge','intensity_rain','intensity_flood','intensity_drought'])


# In[68]:


column_indices = {name: i for i, name in enumerate(train_df.columns)}


# In[69]:


print(column_indices['precipitation_sum (mm)'])
print(column_indices['rain_sum (mm)'])
print(column_indices['river_discharge'])
print(column_indices['intensity_rain'])
print(column_indices['intensity_flood'])
print(column_indices['intensity_drought'])


# In[70]:


mo_baseline_last = Baseline(label_index=[7,8,17,18,19,20])

mo_baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

mo_val_performance = {}
mo_performance = {}

mo_val_performance['Baseline - Last'] = mo_baseline_last.evaluate(mo_wide_window.val)
mo_performance['Baseline - Last'] = mo_baseline_last.evaluate(mo_wide_window.test, verbose=0)


# In[41]:


mo_wide_window.plot(mo_baseline_last)

plt.savefig('fig/mo_baseline_last_rain_sum.png', dpi=300)


# In[43]:


mo_wide_window.plot(model=mo_baseline_last, plot_col='precipitation_sum (mm)')

plt.savefig('fig/mo_baseline_last_precipitation_sum.png', dpi=300)


# In[44]:


mo_wide_window.plot(model=mo_baseline_last, plot_col='river_discharge')

plt.savefig('fig/mo_baseline_last_river_discharge.png', dpi=300)


# In[45]:


mo_wide_window.plot(model=mo_baseline_last, plot_col='intensity_rain')

plt.savefig('fig/mo_baseline_last_intensity_rain.png', dpi=300)


# In[46]:


mo_wide_window.plot(model=mo_baseline_last, plot_col='intensity_flood')

plt.savefig('fig/mo_baseline_last_intensity_flood.png', dpi=300)


# In[47]:


mo_wide_window.plot(model=mo_baseline_last, plot_col='intensity_drought')

plt.savefig('fig/mo_baseline_last_intensity_drought.png', dpi=300)


# In[48]:


print(mo_performance['Baseline - Last'][1])


# Implementing a deep neural network as a multi-output model

# In[49]:


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


# In[50]:


mo_dense = Sequential([
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=6,activation='relu')
])

history = compile_and_fit(mo_dense, mo_single_step_window)

mo_val_performance['Dense'] = mo_dense.evaluate(mo_single_step_window.val)
mo_performance['Dense'] = mo_dense.evaluate(mo_single_step_window.test, verbose=0)


# In[51]:
#scaler = MinMaxScaler()



#X_scaled = scaler.fit_transform(predicted_array)

#obj = scaler.fit(predicted_array)
#X_hat = obj.inverse_transform(X_scaled)
#prediction_copies = np.repeat(predicted_results, 6, axis=-1)
#y_pred_future = scaler.inverse_transform(predicted_array)

mo_mae_val = [v[1] for v in mo_val_performance.values()]
mo_mae_test = [v[1] for v in mo_performance.values()]

x = np.arange(len(mo_performance))

fig, ax = plt.subplots()
ax.bar(x - 0.15, mo_mae_val, width=0.25, color='black', edgecolor='black', label='Validation')
ax.bar(x + 0.15, mo_mae_test, width=0.25, color='white', edgecolor='black', hatch='/', label='Test')
ax.set_ylabel('Mean absolute error')
ax.set_xlabel('Models')

for index, value in enumerate(mo_mae_val):
    plt.text(x=index - 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')

for index, value in enumerate(mo_mae_test):
    plt.text(x=index + 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')

plt.ylim(0, 0.06)
plt.xticks(ticks=x, labels=mo_performance.keys())
plt.legend(loc='best')
plt.tight_layout()

plt.savefig('fig/validation_test.png', dpi=300)


# **LSTM**

# In[52]:


mo_lstm_model = Sequential([
    LSTM(32, return_sequences=True),
    Dense(units = 6,activation='relu')
])

history = compile_and_fit(mo_lstm_model, mo_wide_window)

mo_val_performance = {}
mo_performance = {}

mo_val_performance['LSTM'] = mo_lstm_model.evaluate(mo_wide_window.val)
mo_performance['LSTM'] = mo_lstm_model.evaluate(mo_wide_window.test, verbose=0)


# In[53]:
custom_mo_wide_window = DataWindow(input_width=14, label_width=14, shift=14, label_columns=['precipitation_sum (mm)','rain_sum (mm)','river_discharge','intensity_rain','intensity_flood','intensity_drought'])
input_indices = custom_mo_wide_window.input_indices
label_indices = custom_mo_wide_window.label_indices


predicted_results = mo_lstm_model.predict(custom_mo_wide_window.test)
predicted_array= predicted_results[0]

my_array = np.array(predicted_array)

df = pd.DataFrame(my_array)

df2 =df.rename(columns={0: "precipitation_sum (mm)", 1: "rain_sum (mm)",2:"river_discharge",3:"intensity_rain",4:"intensity_flood",5:"intensity_drought"})

df2.head(14)

mo_wide_window.plot(mo_lstm_model)

plt.savefig('fig/mo_lstm_model.png', dpi=300)


# In[54]:


mo_wide_window.plot(model=mo_lstm_model, plot_col='precipitation_sum (mm)')

plt.savefig('fig/mo_lstm_model_precipitation_sum.png', dpi=300)


# In[55]:


mo_wide_window.plot(model=mo_lstm_model, plot_col='river_discharge')

plt.savefig('fig/mo_lstm_model_river_discharge.png', dpi=300)


# In[56]:


mo_wide_window.plot(model=mo_lstm_model, plot_col='intensity_rain')

plt.savefig('fig/mo_lstm_model_intensity_rain.png', dpi=300)


# In[57]:


mo_wide_window.plot(model=mo_lstm_model, plot_col='intensity_flood')

plt.savefig('fig/mo_lstm_model_intensity_flood.png', dpi=300)


# In[58]:


mo_wide_window.plot(model=mo_lstm_model, plot_col='intensity_drought')

plt.savefig('fig/mo_lstm_model_intensity_drought.png', dpi=300)


# In[ ]:




