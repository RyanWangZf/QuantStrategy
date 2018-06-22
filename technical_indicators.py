# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pdb 
import matplotlib.pyplot as plt 
import models
from fbprophet import Prophet


# ---
# several technical indicators 
# ---

def getATR(df,N=5):
    ''' 
    ATR's corrcoef with range_ts : 0.54191518
    '''
    tr_df = pd.DataFrame({"H_L":(df.high-df.close).values,\
                          "C_H":(df.close.shift(1) - df.high).values,\
                          "C_L":(df.close.shift(1) - df.low).values})
    
    tr_ts = abs(tr_df).max(axis=1)
    atr_ts = tr_ts.rolling(N).mean() # SMA(tr,N)
    df["ATR"] = atr_ts.values
    return df

def getEMV(df,N=5,M=5):
    ''' 
    EMV's corrcoef with range_ts : -0.38
    '''
    a = (df.high + df.low)/2
    b = (df.shift(1).high + df.shift(1).low)/2
    c = df.high - df.low
    vol = df.volume
    em = (a-b) * c / vol 
    emv = em.rolling(N).sum()
    maemv = emv.rolling(M).mean()
    df["emv"] = emv.values
    df["ma_emv"] = maemv.values
    return df

def getFI(df,N=5):
    ''' 
    FI's corrcoef with range_ts : -0.1682
    '''
    fi = df.volume * df.close.diff()
    mafi = fi.rolling(N).mean()
    df["fi"] = mafi.values
    return df

def getMFI(df,N=5):
    '''
    MFI's corrcoef with range_ts : -0.1467
    '''
    typical_price = df.poc
    mf = typical_price.diff() * df.volume
    mf.fillna(0.0,inplace=True)
    mi_list = [np.nan] * N
    for i in range(len(mf)):
        if i >= N:
            this_window = mf[i-N:i]
            mr = this_window.loc[this_window>0].sum() / abs(this_window).sum()
            mfi = 100 - 100/(1+mr)
            mi_list.append(mfi)
    df["mfi"] = mi_list
    return df

def getstd(df,N=5):
    '''
    standard deviation 's corrcoef with range_ts : 0.43474
    '''
    std_list = [np.nan] * N
    for i in range(len(df)):
        if i >= N:
            std_list.append(df.poc[i-N:i].std())
    df["price_std"] = std_list
    return df

def getRSI(df,N=5):
    '''
    RSI's corrcoef with range_ts : -0.19960164
    '''
    rsi_list = [np.nan] * N
    for i in range(len(df)):
        if i >= N:
            this_window = df.poc[i-N:i]
            var = this_window.diff()
            a = var.loc[var>=0].sum()
            b = abs(var.loc[var<=0]).sum()
            rsi_ = 100 * a/(a+b)
            rsi_list.append(rsi_)
    df["rsi"] = rsi_list
    return df

def getKDJ(df,N=5):
    '''
    stochastic oscillator's corrcoef with the range_ts:
    k : -0.1272168
    d : -0.107275
    j : -0.074143
    '''
    num_nan = N - 1
    k_list,d_list,j_list,rsv_list = [np.nan]*num_nan,[np.nan]*num_nan,[np.nan]*num_nan,\
        [np.nan]*num_nan
    k_list[-1] = 50 # initialize value k(0) with 50
    d_list[-1] = 50 # initialize
    j_list[-1] = 100 # initialize
    for i in range(len(df)+1):
        if i >= N:
            this_window = df.iloc[i-N:i,:]
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

# --- 
# indicators' test main function
# ---

if __name__ == "__main__":

    start_date = "20160101"
    end_date = "20160601"

    dbloader = models.basemodel()

    df = dbloader._load_bar("000001",start_date,end_date)

    # test the indicator's corrcoef with the range_ts
    df = getKDJ(df) # get index

    indicator_ar = df["kdj_d"].dropna().values # get values array

    range_ts = (df.high-df.low).iloc[len(df) - len(indicator_ar):]
    corr_ = np.corrcoef(indicator_ar,range_ts)[0][1]
    print("corr: {}".format(corr_))

    pdb.set_trace()

    
