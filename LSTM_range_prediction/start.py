# -*- coding: utf-8 -*-
# Python 3.5 on Linux
# Require: LightGBM, Sklearn, Keras, Numpy, Pandas & Matplotlib.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pdb 
import matplotlib.pyplot as plt 
import datetime
import models

# ---
# several technical indicators 
# ---

class Feature_Generator(object):
    
    def __init__(self):
        self.dbloader = models.basemodel()
    
    def getWeekday(self,df):
        week_day = df.reset_index()["index"].apply(lambda x:\ 
            datetime.datetime.strptime(x,"%Y%m%d").weekday()).values
        df["weekday"] = week_day
        return df

    def getATR(self,df,n=5):
        ''' 
        ATR's corrcoef with range_ts : 0.54191518
        '''
        tr_df = pd.DataFrame({"H_L":(df.high-df.close).values,\
                          "C_H":(df.close.shift(1) - df.high).values,\
                          "C_L":(df.close.shift(1) - df.low).values})
    
        tr_ts = abs(tr_df).max(axis=1)
        atr_ts = tr_ts.rolling(n).mean() # SMA(tr,n)
        df["ATR"] = atr_ts.values
        return df

    def getEMV(self,df,n=5,m=5):
        ''' 
        EMV's corrcoef with range_ts : 
        EMV: -0.38
        MA_EMV: -0.301
        '''
        a = (df.high + df.low)/2
        b = (df.shift(1).high + df.shift(1).low)/2
        c = df.high - df.low
        vol = df.volume
        em = (a-b) * c / vol
        emv = em.rolling(n).sum()
        maemv = emv.rolling(m).mean()
        df["emv"] = emv.values
        df["ma_emv"] = maemv.values
        return df

    def getFI(self,df,n=5):
        '''
        FI's corrcoef with range_ts : -0.1682
        '''
        fi = df.volume * df.close.diff()
        mafi = fi.rolling(n).mean()
        df["fi"] = mafi.values
        return df

    def getMFI(self,df,n=5):
        '''
        MFI's corrcoef with range_ts : -0.1467
        '''
        typical_price = df.poc
        mf = typical_price.diff() * df.volume
        mf.fillna(0.0,inplace=True)
        mi_list = [np.nan] * n
        for i in range(len(mf)):
            if i >= n:
                this_window = mf[i-n:i]
                mr = this_window.loc[this_window>0].sum() / abs(this_window).sum()
                mfi = 100 - 100/(1+mr)
                mi_list.append(mfi)
        df["mfi"] = mi_list
        return df

    def getstd(self,df,n=5):
        '''
        standard deviation 's corrcoef with range_ts : 0.43474
        '''
        std_list = [np.nan] * (n-1)
        for i in range(len(df)):
            if len(df.iloc[:i+1]) < n:
                continue

            std_list.append(df.poc[i-n+1:i+1].std())

        df["price_std"] = std_list
        return df

    def getRSI(self,df,n=5):
        '''
        RSI's corrcoef with range_ts : -0.19960164
        '''
        rsi_list = [np.nan] * n
        for i in range(len(df)):
            if i >= n:
                this_window = df.poc[i-n:i]
                var = this_window.diff()
                a = var.loc[var>=0].sum()
                b = abs(var.loc[var<=0]).sum()
                rsi_ = 100 * a/(a+b)
                rsi_list.append(rsi_)
        df["rsi"] = rsi_list
        return df

    def getKDJ(self,df,n=5):
        '''
        stochastic oscillator's corrcoef with the range_ts
        k : -0.1272168
        d : -0.107275
        j : -0.074143
        '''
        num_nan = n - 1
        k_list,d_list,j_list,rsv_list = [np.nan]*num_nan,[np.nan]*num_nan,[np.nan]*num_nan,\
            [np.nan]*num_nan
        k_list[-1] = 50 # initialize value k(0) with 50
        d_list[-1] = 50 # initialize
        j_list[-1] = 100 # initialize
        for i in range(len(df)+1):
            if i >= n:
                this_window = df.iloc[i-n:i,:]
                ct = df.close[i-1]
                ln = this_window.low.min()
                hn = this_window.high.max()
                rsv_list.append(100*(ct-ln)/(hn-ln))
                k_list.append(k_list[i-2]*2/3 + rsv_list[i-1] * 1/3)
                d_list.append(d_list[i-2]*2/3 + k_list[i-1] * 1/3)
                j_list.append(3*k_list[i-1] - 2*d_list[i-1])

        df["kdj_k"] = k_list
        df["kdj_d"] = d_list
        df["kdj_j"] = j_list

        return df


    def getUOS(self,df,m=7,n=14,o=28):
        '''
        ultimate oscillator's corrcoef with ts_range: 0.17883636
        '''
        preclose = df.close.shift(1)
        bp = pd.Series([np.float32(df.close[i] - min(df.low[i],preclose[i])) for i in range(len(df))])
        tr = pd.Series([max(df.high[i],preclose[i])-min(df.low[i],preclose[i]) for i in range(len(df))])
        avg_m = bp.rolling(m).sum() / tr.rolling(m).sum()
        avg_n = bp.rolling(n).sum() / tr.rolling(n).sum()
        avg_o = bp.rolling(o).sum() / tr.rolling(o).sum()
        ultosc = 100*(4*avg_m + 2*avg_n + avg_o)/(4+2+1)#the original one is (4*avg7+2*avg14+1*avg28)/(7) 
        df["uos"] = ultosc.values
        return df

    def getADX(self,df,n=10,m=5):
        '''
        average directional indicator(ADX)'s corrcoef with range_ts:
        -di: 0.11329
        +di: 0.02433
        adx: -0.162109
        '''

        pdm = (df.high - df.high.shift(1)).apply(lambda x: max(x,0))
        ndm = (df.low.shift(1) - df.low).apply(lambda x:max(x,0))
        ndm[ndm<pdm] = 0
        pdm[pdm<ndm] = 0
        tr_df = pd.DataFrame({"H_L":(df.high-df.close).values,\
                               "C_H":(df.close.shift(1) - df.high).values,\
                                "C_L":(df.close.shift(1) - df.low).values})
        tr_ts = abs(tr_df).max(axis=1)
        tr_ts.index = df.index

        trn = tr_ts.rolling(n).sum()
        pdmn = pdm.rolling(n).sum()
        ndmn = ndm.rolling(n).sum()

        pdi = (pdmn/trn) * 100
        ndi = (ndmn/trn) * 100
        dx = 100 * abs(pdi - ndi) / abs(pdi + ndi)
        adx = dx.rolling(m).mean()

        df["adx"] = adx.values
        df["ndi"] = ndi.values
        df["pdi"] = pdi.values

        return df

    def getTRd(self,df,n=5):
        '''
        in-day true range 's corr with range_ts:
        trm(morning): 0.7649577
        tra(afternoon): 0.7739293
        '''
        sid = df.sid.iloc[0]
        date_list = df.index.tolist()
        tra,trm = [],[]
        for i,date_ in enumerate(date_list):
            if i < 1:
                continue
            this_day = self.dbloader._load_tick(sid,date_,date_)
            morning_ = this_day.between_time("9:30","11:30")
            afternoon_= this_day.between_time("13:00","14:55")
            if len(morning_) == 0 or len(afternoon_) == 0: # ticks are missing for unknown reason
                tra.append(np.nan)
                trm.append(np.nan)
                continue

            close_m = df.iloc[i-1].close
            close_a = afternoon_.AskPrice1.iloc[-1]
            trm.append(max(abs(close_m-morning_.AskPrice1.max()),abs(close_m-morning_.AskPrice1.min()),\
                abs(morning_.AskPrice1.max()-morning_.AskPrice1.min())))
            tra.append(max(abs(close_a-afternoon_.AskPrice1.max()),\
                abs(close_a-afternoon_.AskPrice1.min()),\
                abs(afternoon_.AskPrice1.max()-afternoon_.AskPrice1.min())))

        tra,trm = [np.nan] + tra, [np.nan] + trm
        df["TRm"] = trm
        df["TRa"] = tra
        for col in ["TRm","TRa"]:
            df[col] = df[col].fillna(method="ffill")

        return df

    def getVWATR(self,df,n=5):
        '''
        volume weighted atr 's corr with range_ts:
        vwatr: 0.6598347
        '''
        tr_df = pd.DataFrame({"H_L":(df.high-df.close).values,\
            "C_H":(df.close.shift(1) - df.high).values,\
            "C_L":(df.close.shift(1) - df.low).values})
        tr_ts = abs(tr_df).max(axis=1)
        vwatr = []
        for i,date_ in enumerate(df.index):
            if len(tr_ts[:i+1]) < 5:
                continue
            print(date_)
            vwatr.append((tr_ts[i-n+1:i+1].values * df.iloc[i-n+1:i+1].volume.values).mean())
        vwatr = [np.nan] * (n-1) + vwatr
        df["vwatr"] = vwatr
        return df

    def getVolstd(self,df,n=5):
        '''
        volume std 's corr with range_ts:
        volstd: 0.271973
        '''
        std_list = [np.nan] * (n-1)
        for i in range(len(df)):
            if df.iloc[:i+1].shape[0] < n:
                continue
            std_list.append(df.volume[i-n+1:i+1].std())
        df["vol_std"] = std_list
        return df

    def getCumsum(self,df,n=5):
        '''
        cumsum 's corr with range_ts:
        cumprice(minutes): 0.5625
        cumvol(minutes): 0.4649
        '''
        sid = df.sid.iloc[0]
        date_list = df.index.tolist()
        cum_price,cum_vol = [],[]
        for date_ in date_list:
            this_day = self.dbloader._load_tick(sid,date_,date_)
            this_day = this_day.resample("min").last().fillna(method="ffill")
            askprice = this_day.AskPrice1.copy()
            vol = this_day.Volume.copy()
            cum_price.append(float(askprice.diff().abs().sum()))
            cum_vol.append(float(vol.diff().abs().sum()))

        df["cumprice"] = cum_price
        df["cumvol"] = cum_vol
        return df

    def getRnglast(self,df,n=1):
        '''
        lasy n days range
        '''
        range_ts = df["high"] - df["low"]
        df["range_last"] = range_ts.diff(n).values
        return df

# ---
# LSTM functions
# ---

class LSTM_model(object):

    def __init__(self,features=["high","volume"],timesteps=3):
        self.features = features
        self.timesteps = timesteps

    def createDataset(self,df,standardize=False):
        timesteps = self.timesteps
        datasety = df["high"] - df["low"] # target is the every day range
        df = df[self.features] # selected features
        df = df.shift(1).dropna(how="all")
        if standardize: # standardize the raw dataset and return the scalerx
            scalerx = StandardScaler()
            ind_list = df.index.copy()
            df = scalerx.fit_transform(df)
            df = pd.DataFrame(df,index=ind_list)

        for counter,i in enumerate(range(timesteps,len(df))):
            this_ = df.iloc[i-timesteps:i]
            if counter == 0:
                datasetx = this_.values.reshape(1,this_.shape[0],this_.shape[1])
            else:
                datasetx = np.r_[datasetx,this_.values.reshape(1,this_.shape[0],this_.shape[1])]
        datasety = datasety[1+timesteps:]
        if not standardize:
            return datasetx,datasety
        else:
            return datasetx,datasety,scalerx

    def createLSTM(self):
        data_dim = len(self.features)
        timesteps = self.timesteps
        from keras.layers import LSTM,Dense
        from keras.models import Sequential
        from sklearn.metrics import mean_squared_error

        model = Sequential()
        model.add(LSTM(32,return_sequences=True,input_shape=(timesteps,data_dim)))
        model.add(LSTM(64,return_sequences=True))
        model.add(LSTM(128,return_sequences=False))

        model.add(Dense(1))
        model.compile(loss="mean_squared_error",optimizer="adam")
        return model

    def train(self,df):
        timesteps = self.timesteps
        # drop NaN
        df = df.dropna()
        # split train & test sets
        num_train =  int(0.7*len(df))
        df_train = df.iloc[:num_train]
        trainy = df_train["high"]-df_train["low"]
        df_test = df.iloc[num_train:]
        scalerx = StandardScaler()
        scalerx.fit(df_train)
        # create datasets and training
        dataset = pd.DataFrame(scalerx.transform(df),columns=df.columns)
        datasetx,_ = self.createDataset(dataset,standardize=False)
        datasety = (df["high"]-df["low"]).values.reshape(-1,1)[1+timesteps:]
        trainx,trainy = datasetx[:num_train],datasety[:num_train]
        testx,testy = datasetx[num_train:],datasety[num_train:]
        model_lstm = self.createLSTM()
        model_lstm.fit(trainx,trainy,epochs=50,validation_data=(testx,testy),batch_size=1,verbose=2)
        # save results
        self.estimator_ = model_lstm
        self.trainx_ = trainx
        self.trainy_ = trainy
        self.testx_ = testx
        self.testy_ = testy

    def plot_result(self):
        plt.subplot(211)
        self.fity_ = self.estimator_.predict(self.trainx_)
        plt.plot(self.trainy_,label="real")
        plt.plot(self.fity_,label="fitted")
        plt.legend()
        plt.subplot(212)
        self.predy_ = self.estimator_.predict(self.testx_)
        plt.plot(self.testy_,label="real")
        plt.plot(self.predy_,label="pred")
        plt.legend()
        plt.show()

# ---
# Features assessment & selection
# ---

def plot_feature_importance(df):
    '''
    select high,volume,ATR,weekday and EMV
    '''
    import lightgbm as lgb
    x,y = createXY(df)
    reg = lgb.LGBMRegressor(boosting_type="gbdt",num_leaves=2**8,max_depth=8,learning_rate=0.1,\
        n_estimators=100,min_child_samples=1,reg_alpha=0.0,reg_lambda=0.0,random_state=84, \
        n_jobs=4,silent=True)
    reg.fit(x,y)
    lgb.plot_importance(reg)
    plt.show()

def createXY(df):
    y = (df.high - df.low).shift(-1)
    x = df.loc[y.notnull()]
    y = y[x.dropna().index]
    x = x.dropna()
    x = x.drop(["sid"],axis=1)
    return x,y

# --- 
# indicators' test main function
# ---

def print_corr(df,var):

    indicator_ar = df[var].dropna().values
    range_ts = (df.high-df.low).iloc[len(df) - len(indicator_ar):]
    try:
        corr_ = np.corrcoef(indicator_ar,range_ts)[0][1]
        print("{} corr: {}".format(var,corr_))
    except:
        print("unable to process {}'s dtype: {} ".format(var,indicator_ar.dtype))

def print_corr_matrix(df,select_cols):

    df_corr = df.dropna()
    df_corr = df[select_cols]
    correlation = df_corr.corr()
    names = df_corr.columns.tolist()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlation,vmin=-1,vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(names),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()
# ---
# main script
# ---

if __name__ == "__main__":

    np.random.seed(84)

    sid = "000001"
    start_date = "20160101"
    end_date = "20160601"
    dbloader = models.basemodel()

    df = dbloader._load_bar(sid,start_date,end_date)
    fg = Feature_Generator()
    '''
    # generate features
    df = fg.getWeekday(df)
    df = fg.getATR(df)
    df = fg.getEMV(df)
    df = fg.getRnglast(df)

    # select feature & training
    selected_ft = ["high","volume","weekday","ATR","emv","range_last"]
    lstm = LSTM_model(features=selected_ft,timesteps=1)
    lstm.train(df)
    lstm.plot_result()
    '''

    ''' 
    # plot feature importances by lightgbm regressor
    plot_feature_importance(df)
    '''


    # print correlation
    for col in df.columns:
        print_corr(df,col)

    '''
    # plot correlation heat matrix
    select_cols = ["ATR","price_std","emv","rsi"]
    print_corr_matrix(df,select_cols)
    
    # print correlation coefficients
    for col in df.columns:
        print_corr(df,col)
    '''
    correlation_with_range_ts_dict = { \
        "ATR":0.5419,   "emv":-0.38,    "ma_emv":-0.301, \
        "fi":-0.1682,   "mfi":-0.1467,  "price_std":0.43474, \
        "rsi":-0.1996,  "kdj_k":-0.1272,"kdj_d":-0.1072, \
        "kdj_j":-0.0741,"uos":0.17884,"ndi":0.11329, \
        "pdi": 0.02433, "adx":-0.1621, \
        }
    pdb.set_trace()
