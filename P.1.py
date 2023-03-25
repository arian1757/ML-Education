#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from torch import nn

# from keras.utils import FeatureSpace


# In[44]:


"""
Forcasting forest cover type from cartographic variables
elevational value is a important feature 



"""


# In[45]:


df=pd.read_csv('./ML-EX/happy_new_year_1402/1/covtype.data')


# In[46]:


# pd.set_option('display.max_columns',None)
# df.describe()


# In[47]:


# categorizing samples
test_dataframe=df.sample(frac=0.2,random_state=43)
df_train_val=df.drop(test_dataframe.index)
val_dataframe=df_train_val.sample(frac=0.2,random_state=85)
train_dataframe=df_train_val.drop(val_dataframe.index)


# In[48]:


# def dataframe_to_dataset(dataframe):
#     dataframe=dataframe.copy()
#     Cover_Type=dataframe.pop('Cover_Type')
#     ds = tf.data.Dataset.from_tensor_slices((dict(dataframe),Cover_Type))
#     ds = ds.shuffle(buffer_size=len(dataframe))
#     return ds
# test_ds = dataframe_to_dataset(test_dataframe)
# val_ds = dataframe_to_dataset(val_dataframe)
# train_ds = dataframe_to_dataset(val_dataframe)

# feature_space = tf.keras.utils.FeatureSpace(
#     features={
#         # The features should be categorical
#         "Wilderness_Area1": "integer_categorical",
#         "Wilderness_Area2": "integer_categorical",
#         "Wilderness_Area3": "integer_categorical",
#         "Wilderness_Area4": "integer_categorical",
#         "Soil_Type1": "integer_categorical",
#         "Soil_Type2": "integer_categorical",
#         "Soil_Type3": "integer_categorical",
#         "Soil_Type4": "integer_categorical",
#         "Soil_Type5": "integer_categorical",
#         "Soil_Type6": "integer_categorical",
#         "Soil_Type7": "integer_categorical",
#         "Soil_Type8": "integer_categorical",
#         "Soil_Type9": "integer_categorical",
#         "Soil_Type10": "integer_categorical",
#         "Soil_Type11": "integer_categorical",
#         "Soil_Type12": "integer_categorical",
#         "Soil_Type13": "integer_categorical",
#         "Soil_Type14": "integer_categorical",
#         "Soil_Type15": "integer_categorical",
#         "Soil_Type16": "integer_categorical",
#         "Soil_Type17": "integer_categorical",
#         "Soil_Type18": "integer_categorical",
#         "Soil_Type19": "integer_categorical",
#         "Soil_Type20": "integer_categorical",
#         "Soil_Type21": "integer_categorical",
#         "Soil_Type22": "integer_categorical",
#         "Soil_Type23": "integer_categorical",
#         "Soil_Type24": "integer_categorical",
#         "Soil_Type25": "integer_categorical",
#         "Soil_Type26": "integer_categorical",
#         "Soil_Type27": "integer_categorical",
#         "Soil_Type28": "integer_categorical",
#         "Soil_Type29": "integer_categorical",
#         "Soil_Type30": "integer_categorical",
#         "Soil_Type31": "integer_categorical",
#         "Soil_Type32": "integer_categorical",
#         "Soil_Type33": "integer_categorical",
#         "Soil_Type34": "integer_categorical",
#         "Soil_Type35": "integer_categorical",
#         "Soil_Type36": "integer_categorical",
#         "Soil_Type37": "integer_categorical",
#         "Soil_Type38": "integer_categorical",
#         "Soil_Type39": "integer_categorical",
#         "Soil_Type40": "integer_categorical",
        
        
#         # Numerical features to normalize
#         "Elevation": "float_normalized",
#         "Aspect": "float_normalized",
#         "Slope": "float_normalized",
#         "Horizontal_Distance_To_Hydrology": "float_normalized",
#         "Vertical_Distance_To_Hydrology ": "float_normalized",
#         "Horizontal_Distance_To_Roadways": "float_normalized",
#         "Hillshade_9am": "float_normalized",
#         "Hillshade_Noon": "float_normalized",
#         "Hillshade_3pm": "float_normalized",
#         "Horizontal_Distance_To_Fire_Points": "float_normalized",
        
#     },
    
#     output_mode="concat",
# )
# train_ds_with_no_labels = train_ds.map(lambda x, _: x)
# feature_space.adapt(train_ds_with_no_labels)


# In[49]:


# normalizing data (just for the 10 features, the rest of them are binary)
scaler = StandardScaler()
X_train=scaler.fit_transform(train_dataframe.iloc[:,:10])


# In[50]:


# combine normalized features with others
X_train = np.concatenate((X_train,train_dataframe.iloc[:,10:].values),axis=1)


# In[51]:


def normalization(data):
    data_scaled= scaler.transform(data.iloc[:,:10])
    data=np.concatenate((data_scaled,data.iloc[:,10:].values),axis=1)
    
    return data



# In[52]:


X_val=normalization(val_dataframe)


# In[53]:


# prepare labels
def to_categorical(data):
    data=tf.keras.utils.to_categorical(data)
    
    return data
# making dataset
def dataset(x,y):
    x=x.astype(np.float32)
    y=y.astype(np.float32)
    x=torch.from_numpy(x)
    y= torch.from_numpy(y)
    ds= torch.utils.data.TensorDataset(x,y)
    
    return ds


# In[54]:


Y_train= train_dataframe.iloc[:,-1]
Y_train =to_categorical(Y_train)
Y_val= val_dataframe.iloc[:,-1]
Y_val =to_categorical(Y_val)
train_ds=dataset(X_train,Y_train)


# In[55]:


val_ds=dataset(X_val,Y_val)


# In[56]:


train_ds = torch.utils.data.DataLoader(train_ds, batch_size=128,shuffle=True)
val_ds = torch.utils.data.DataLoader(val_ds, batch_size=128,shuffle=False)


# In[57]:


class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu = nn.Sequential(nn.Linear(55,300),
                                         nn.ReLU(),
                                        #  nn.Dropout(p=0.1),
                                        nn.BatchNorm1d(300),
                                        nn.Linear(300,450),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(450),
                                        nn.Linear(450,8),
                                        nn.Softmax(dim=1)
                                         


                                         )
    def forward(self,x):
        x= self.linear_relu(x)
        
        return x
model =network()


# In[58]:


# learning rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
epochs = 5


# In[59]:


# device = torch.device("mps")
# model.to(device=device)


# In[60]:


for epoch in range(epochs):
    running_loss=0.0
    correct=0
    for data in  iter(train_ds):
        inputs,labels = data
        # inputs=inputs.to('mps')
        # labels=labels.to(device)
        
        
        
        y_pred=model(inputs)
        loss = criterion(y_pred,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
        correct += y_pred.argmax(1).eq(labels.argmax(1)).sum()
   

    
    alpha = len(train_ds.dataset)/128
    train_loss = running_loss/alpha
    accuracy = correct /len(train_ds.dataset)

    print(f'loss in epoch {epoch+1} is {train_loss} and accuracy is {accuracy} ')
    running_loss=0
    correct=0
"""" The Result (Train):

            loss in epoch 1 is 1.3131907630845399 and accuracy is 0.961236834526062 
            loss in epoch 2 is 1.3104638722397313 and accuracy is 0.9639261364936829 
            loss in epoch 3 is 1.3106455412777054 and accuracy is 0.9638158679008484 
            loss in epoch 4 is 1.310657603808246 and accuracy is 0.9637943506240845 
            loss in epoch 5 is 1.3119194241845846 and accuracy is 0.9625680446624756 
"""

# In[61]:


def test (dataloader):
    running_loss=0.0
    correct=0
    with torch.no_grad():
        for x,y in dataloader:
            pred= model(x)
            running_loss += criterion(pred,y).item()
            correct += pred.argmax(1).eq(y.argmax(1)).sum()
    accuracy = correct /len(dataloader.dataset) 
    loss_test=running_loss/  len(dataloader)     
    print(f'loss  is {loss_test} and accuracy is {accuracy} ')




# In[62]:


test(val_ds)

"""" The Result (Validation):

            loss  is 1.3092541551983503 and accuracy is 0.9647812843322754 
"""


# In[63]:


X_test = normalization(test_dataframe)
Y_test= test_dataframe.iloc[:,-1]
Y_test =to_categorical(Y_test)
test_ds=dataset(X_test,Y_test)
test_ds = torch.utils.data.DataLoader(test_ds, batch_size=128,shuffle=False)
test(test_ds)

"""" The Result (Test):

            loss  is 1.3083921002658978 and accuracy is 0.9656116366386414 
"""

