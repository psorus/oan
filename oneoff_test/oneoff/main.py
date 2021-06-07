"""
Title: Timeseries anomaly detection using an Autoencoder
Author: [pavithrasv](https://github.com/pavithrasv)
Date created: 2020/05/31
Last modified: 2020/05/31
Description: Detect anomalies in a timeseries using an Autoencoder.
"""

"""
## Introduction
This script demonstrates how you can use a reconstruction convolutional
autoencoder model to detect anomalies in timeseries data.
"""

"""
## Setup
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
#from matplotlib import pyplot as plt
from plt import *
from time import time
from tqdm import tqdm

"""
## Load the data
We will use the [Numenta Anomaly Benchmark(NAB)](
https://www.kaggle.com/boltzmannbrain/nab) dataset. It provides artifical
timeseries data containing labeled anomalous periods of behavior. Data are
ordered, timestamped, single-valued metrics.
We will use the `art_daily_small_noise.csv` file for training and the
`art_daily_jumpsup.csv` file for testing. The simplicity of this dataset
allows us to demonstrate anomaly detection effectively.
"""

master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
df_small_noise_url = master_url_root + df_small_noise_url_suffix
df_small_noise = pd.read_csv(
    df_small_noise_url, parse_dates=True, index_col="timestamp"
)

df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
df_daily_jumpsup = pd.read_csv(
    df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
)

"""
## Quick look at the data
"""

#print(df_small_noise.head())

#print(df_daily_jumpsup.head())

"""
## Visualize the data
### Timeseries data without anomalies
We will use the following data for training.
"""
#fig, ax = plt.subplots()
#df_small_noise.plot(legend=False, ax=ax)
#plt.show()

"""
### Timeseries data with anomalies
We will use the following data for testing and see if the sudden jump up in the
data is detected as an anomaly.
"""
#fig, ax = plt.subplots()
#df_daily_jumpsup.plot(legend=False, ax=ax)
#plt.show()

"""
## Prepare training data
Get data values from the training timeseries data file and normalize the
`value` data. We have a `value` for every 5 mins for 14 days.
-   24 * 60 / 5 = **288 timesteps per day**
-   288 * 14 = **4032 data points** in total
"""


# Normalize and save the mean and std we get,
# for normalizing test data.
training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
df_training_value = (df_small_noise - training_mean) / training_std
#print("Number of training samples:", len(df_training_value))

"""
### Create sequences
Create sequences combining `TIME_STEPS` contiguous data values from the
training data.
"""

TIME_STEPS = 288

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(df_training_value.values)
#print("Training input shape: ", x_train.shape)

"""
## Build a model
We will build a convolutional reconstruction autoencoder model. The model will
take input of shape `(batch_size, sequence_length, num_features)` and return
output of the same shape. In this case, `sequence_length` is 288 and
`num_features` is 1.
"""

model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu",use_bias=False,
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu",use_bias=False,
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu",use_bias=False,
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu",use_bias=False,
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same",use_bias=False),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

"""
## Train the model
Please note that we are using `x_train` as both the input and the target
since this is a reconstruction model.
"""
t0=time()
history = model.fit(
    x_train,
    np.ones_like(x_train),
    epochs=100,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)
t1=time()

with open("time","w") as f:
  f.write(str(t1-t0))
"""
Let's plot training and validation loss to see how the training went.
"""

#plt.plot(history.history["loss"], label="Training Loss")
#plt.plot(history.history["val_loss"], label="Validation Loss")
#plt.legend()
#plt.show()

"""
## Detecting anomalies
We will detect anomalies by determining how well our model can reconstruct
the input data.
1.   Find MAE loss on training samples.
2.   Find max MAE loss value. This is the worst our model has performed trying
to reconstruct a sample. We will make this the `threshold` for anomaly
detection.
3.   If the reconstruction loss for a sample is greater than this `threshold`
value then we can infer that the model is seeing a pattern that it isn't
familiar with. We will label this sample as an `anomaly`.
"""

# Get train MAE loss.
x_train_pred = model.predict(x_train)
M=np.mean(x_train_pred)
train_mae_loss = np.mean(np.abs(x_train_pred - np.ones_like(x_train)*M), axis=1)

#plt.hist(train_mae_loss, bins=50)
#plt.xlabel("Train MAE loss")
#plt.ylabel("No of samples")
#plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
#print("Reconstruction error threshold: ", threshold)

"""
### Compare recontruction
Just for fun, let's see how our model has recontructed the first sample.
This is the 288 timesteps from day 1 of our training dataset.
"""

# Checking how the first sequence is learnt
#plt.plot(x_train[0])
#plt.plot(x_train_pred[0])
#plt.show()

"""
### Prepare test data
"""


def normalize_test(values, mean, std):
    values -= mean
    values /= std
    return values


df_test_value = (df_daily_jumpsup - training_mean) / training_std
#fig, ax = plt.subplots()
#df_test_value.plot(legend=False, ax=ax)
#plt.show()

# Create sequences from test values.
x_test = create_sequences(df_test_value.values)
#print("Test input shape: ", x_test.shape)

# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - np.ones_like(x_test)*M), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

#plt.hist(test_mae_loss, bins=50)
#plt.xlabel("test MAE loss")
#plt.ylabel("No of samples")
#plt.show()

def bythreshold(threshold):

    # Detect all the samples which are anomalies.
    anomalies = test_mae_loss > threshold
    #print("Number of anomaly samples: ", np.sum(anomalies))
    #print("Indices of anomaly samples: ", np.where(anomalies))

    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_indices = []
    normal_data_indices=[]
    for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
        if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)
        else:
            normal_data_indices.append(data_idx)

    return anomalous_data_indices,normal_data_indices

"""
Let's overlay the anomalies on the original test data plot.
"""
anomalous_data_indices,normal_data_indices=bythreshold(threshold)

def eval(threshold):
    #print(threshold)
    border=100.0

    ano,norm=bythreshold(threshold)
    ano=df_daily_jumpsup.iloc[ano].to_numpy()
    norm=df_daily_jumpsup.iloc[norm].to_numpy()
    tt=np.sum(ano>border)
    ff=np.sum(norm<=border)
    tf=np.sum(ano<=border)
    ft=np.sum(norm>border)

    #print(tt,tf,ft,ff)


    tpr=(tt)/(tt+ft)
    fpr=(tf)/(ff+tf)
    return fpr,tpr

def multi_eval():
    threshs=np.arange(0,threshold,threshold/100)
    fprs,tprs=[],[]
    for t in tqdm(threshs):
        fpr,tpr=eval(t)
        fprs.append(fpr)
        tprs.append(tpr)
    return fprs,tprs


fpr,tpr=multi_eval()

np.savez_compressed("roc",fpr=fpr,tpr=tpr)


df_subset = df_daily_jumpsup.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.savefig("out.png",format="png")
plt.savefig("out.pdf",format="pdf")
plt.show()

print(eval(threshold))

exit()
sizes=[np.max(zw) for zw in x_test]

i=np.argmax(sizes)
i0=i-5
i1=i+5

def plotday(i):
  plt.plot(x_test[i])
  plt.plot(x_test_pred[i])

def plotdays(*ii):
  a=[np.reshape(x_test[i],np.prod(x_test[i].shape)) for i in ii]
  b=[np.reshape(x_test_pred[i],np.prod(x_test_pred[i].shape)) for i in ii]

  a=np.concatenate(a,axis=0)
  b=np.concatenate(b,axis=0)
  plt.plot(a)
  plt.plot(b)

#plotdays(*[i for i in range(i0,i1)])
#plt.show()

