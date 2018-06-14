# -*- coding: utf-8 -*-

import numpy as np
import models
import pandas as pd
import pdb 
import matplotlib.pyplot as plt 
import datetime
from sklearn.preprocessing import StandardScaler


def getEveryday(start_date,end_date):
    start_date = datetime.datetime.strptime(start_date,"%Y%m%d")
    end_date = datetime.datetime.strptime(end_date,"%Y%m%d")
    date_list = []
    while start_date <= end_date:
        date_list.append(start_date.strftime("%Y%m%d"))
        start_date += datetime.timedelta(days=1)
    
    return date_list

def createLSTM(timesteps,data_dim):
    from keras.layers import LSTM,Dense
    from keras.models import Sequential
    from sklearn.metrics import mean_squared_error
    
    model = Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(timesteps,data_dim)))
    model.add(LSTM(64,return_sequences=False))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    return model    

def createDataset(df,timesteps,standardize=False):

    datasety = df["high"] - df["low"]  # use range of every day as label
    df = df.shift(1).dropna(how="all")
    if standardize: # standardize the raw dataset
        scalerx =  StandardScaler()
        scalery = StandardScaler()
        ind_list = df.index.copy()
        df = scalerx.fit_transform(df)
        df = pd.DataFrame(df,index=ind_list)
        datasety = scalery.fit_transform(datasety.reshape(-1,1))

    # ind_ = []
    for counter,i in enumerate(range(timesteps,len(df))):
        # print(counter)
        this_ = df.iloc[i-timesteps:i]
        # ind_.append(df.index[i-1])
        if counter == 0:
            datasetx = this_.values.reshape(1,this_.shape[0],this_.shape[1])
        else:
            datasetx = np.r_[datasetx,this_.values.reshape(1,this_.shape[0],this_.shape[1])]
    datasety = datasety[1+timesteps:]
    if not standardize:
        return datasetx,datasety
    else:
        return datasetx,datasety,scalerx,scalery

if __name__ == "__main__":

    # set random seed
    np.random.seed(84)

    dbloader = models.basemodel()
    sid = "000001"
    start_date = "20160601"
    end_date = "20170201"
    timesteps = 3

    df = dbloader._load_bar(sid,start_date,end_date)

    # split train/test set and get scalerx and scalery
    num_train = int(0.7*len(df))
    df_train = df.iloc[:num_train]
    trainy = df_train["high"] - df_train["low"]
    df_test = df.iloc[num_train:]
    scalerx = StandardScaler()
    scalery = StandardScaler()
    scalerx.fit(df_train)
    scalery.fit(trainy.values.reshape(-1,1))

    # get dataset used for input in LSTM model
    dataset = pd.DataFrame(scalerx.transform(df),columns=df.columns)
    datasetx,_ = createDataset(dataset,timesteps=3,standardize=False)
    datasety = scalery.transform((df["high"] - df["low"]).values.reshape(-1,1))[1+timesteps:]
    trainx,trainy = datasetx[:num_train],datasety[:num_train]
    testx,testy = datasetx[num_train:],datasety[num_train:]

    # create LSTM model
    model_lstm = createLSTM(timesteps=timesteps,data_dim=4)
    model_lstm.fit(trainx,trainy,epochs=50,batch_size=1,verbose=2)


    # observe fitted_y
    pred_y = model_lstm.predict(trainx)
    pred_y = scalery.inverse_transform(pred_y)
    trainy = scalery.inverse_transform(trainy)
    plt.plot(trainy,label="real")
    plt.plot(pred_y,label="fit")
    plt.legend()
    plt.show()

    pdb.set_trace()

    # observe pred_y
    predy = model_lstm.predict(testx)
    predy = scalery.inverse_transform(predy)
    testy = scalery.inverse_transform(testy)
    plt.plot(testy,label="real")
    plt.plot(predy,label="pred")
    plt.legend()
    plt.show()

    pdb.set_trace()

    '''
    range with poc: corrcoef = 0.44477105


    '''
        
        
        
        
        
        
        