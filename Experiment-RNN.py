
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
# get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


n = str(2)
fullDF = pd.read_csv("/home/users/nus/a0112066/data/Chiller"+n+"_full.csv")
num_samples = fullDF.shape[0]
window_size = 60


# In[3]:


fullDF.head()


# In[4]:


Watt = fullDF[["ch1Watt", "ch2Watt", "ch3Watt"]].as_matrix()
Value = fullDF[["value1", "value2", "value3", "value4"]].as_matrix()
Conflow = fullDF[["conflowRate", "conflowSpeed"]].as_matrix()
Evaflow = fullDF[["evaflowRate", "evaflowSpeed"]].as_matrix()
Total_Watt = np.sum(Watt, axis = 1)


# In[5]:


from sklearn import preprocessing


# In[6]:


Watt_scaled = preprocessing.scale(Watt)
Value_scaled = preprocessing.scale(Value)
Conflow_scaled = preprocessing.scale(Conflow)
Evaflow_scaled = preprocessing.scale(Evaflow)
Total_Watt_scaled = preprocessing.scale(Total_Watt)


# In[7]:


Hour = pd.to_datetime(fullDF.ts).dt.hour.as_matrix()
Hour_transformed = np.zeros([num_samples,24])
for i in range(num_samples):
    Hour_transformed[i, Hour[i]] = 1


# In[8]:


Day = pd.to_datetime(fullDF.ts).dt.day.as_matrix()
Day_transformed = np.zeros([num_samples,31])
for i in range(num_samples):
    Day_transformed[i, Day[i] - 1] = 1


# In[9]:


Month = pd.to_datetime(fullDF.ts).dt.month.as_matrix()
Month_transformed = np.zeros([num_samples,12])
for i in range(num_samples):
    Month_transformed[i, Month[i] - 1] = 1


# In[10]:


# Data = np.concatenate([Hour_transformed, Value_scaled, Conflow_scaled, Evaflow_scaled], axis =1)
Data = np.concatenate([Value_scaled, Conflow_scaled, Evaflow_scaled], axis =1)


# In[11]:


dimension = Data.shape[1]


# In[12]:


Data_align = np.zeros((window_size, num_samples - window_size + 1, dimension))
for i in range(0, window_size):
    Data_align[i] = Data[i: num_samples - window_size + 1 + i]


# In[13]:


Full_Data = Data_align.transpose((1,0,2))


# In[14]:


Power = Total_Watt[:num_samples - window_size + 1].reshape((-1,1))


# In[15]:


Moment = np.concatenate([np.zeros((1,1)),Power[1:] - Power[:-1]], axis = 0)


# In[16]:


Force = np.concatenate([np.zeros((1,1)),Moment[1:] - Moment[:-1]], axis = 0)


# In[17]:


import keras
from keras import layers
from keras.layers import Lambda, Activation,recurrent, Bidirectional, Dense, Flatten, Conv1D, Dropout, LSTM, GRU, concatenate, multiply, add, Reshape, MaxPooling1D, BatchNormalization
from keras.models import Model, load_model
import keras.backend as K


# In[18]:


def mean_pred(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)/(y_true + 0.01))


# In[19]:


inp = layers.Input(shape=(window_size, dimension), dtype='float32')


# In[20]:


enc1 = LSTM(50,return_sequences=True)(inp)
enc2 = LSTM(50,return_sequences=True)(enc1)
flatten = Flatten()(enc2)


# In[21]:


dense1 = Dense(500, activation='relu')(flatten)
output1 = Dense(1)(dense1)


# In[22]:


dense2 = Dense(500, activation='relu')(flatten)
output2 = Dense(1)(dense2)


# In[23]:


dense3 = Dense(500, activation='relu')(flatten)
output3 = Dense(1)(dense3)


# In[24]:


model = Model([inp],[output1])
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=[mean_pred])


# In[25]:

val = .1
train_num = int(len(Full_Data) * val)
test_inp = Full_Data[:train_num]
test_out1 = Power[:train_num]
test_out2 = Moment[:train_num]
test_out3 = Force[:train_num]
val_inp = Full_Data[train_num:]
val_out1 = Power[train_num:]
val_out2 = Moment[train_num:]
val_out3 = Force[train_num:]


# In[26]:


history = model.fit([test_inp], [test_out1],
          epochs=10,
          validation_data=([val_inp], [val_out1]))


# In[27]:


# model.summary()


# In[28]:


# pred = model.predict(val_inp, verbose=1)

with open("/home/users/nus/a0112066/data/history", "wb") as h:
    pickle.dump(history.history, h)
model.save("/home/users/nus/a0112066/data/model-rnn")
# In[29]:

#
# pred
#
#
# # In[30]:
#
#
# a = (pred[0] - val_out1)/val_out1
#
#
# # In[39]:
#
#
# b = a > .1
#
#
# # In[40]:
#
#
# np.sum(b)
