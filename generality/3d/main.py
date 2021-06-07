from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

from loss import loss

from data import load_train,load_anomaly

data=load_train()
anoma=load_anomaly()


fdim=3



model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
#model.add(layers.Embedding(input_dim=1, output_dim=8))

model.add(layers.Input(data.shape[1:]))

model.add(layers.Dense(5,use_bias=False))

model.add(layers.LSTM(fdim*8,return_sequences=True))
model.add(layers.LSTM(fdim*3,return_sequences=True))
model.add(layers.Dense(fdim,use_bias=False))
#model.add(layers.SimpleRNN(fdim,return_sequences=True))

#model.add(layers.Reshape((1,1,-1,fdim)))

#model.add(layers.Dense(1))


model.summary()

model.compile(optimizer="adam",loss=loss(fdim,self=True))


model.fit(data,data,
        batch_size=50,
        epochs=50,
        validation_split=0.2,
        callbacks=[keras.callbacks.EarlyStopping(patience=15)])

p=model.predict(anoma)

np.savez_compressed("output",x=anoma,y=p)




