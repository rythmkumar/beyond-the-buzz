import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('train.csv')
X_train=dataset.iloc[:,1:]
Y_train=dataset.iloc[:,:1]

test_data=pd.read_csv('test.csv')
X_test=test_data.iloc[:,1:]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([Dense(units=9,activation='relu'), Dense(units=7,activation='relu'), Dense(units=7,activation='relu'), Dense(1,activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Early stoping
early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=10,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)
model_history=model.fit(X_train,Y_train,validation_split=0.33,batch_size=10,epochs=200,callbacks=early_stopping)

Y_pred=model.predict(X_test)
Y_pred=[1 if(i>=0.5) else 0 for i in Y_pred]

Predictions = pd.DataFrame(Y_pred)
Predictions.to_csv('predictions.csv')
