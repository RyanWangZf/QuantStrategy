# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from multiprocessing import Pool,Manager

import os
import time
import pdb 

def operation(position,cash,trans_record,tag="long",change_position=100):
    if tag == "long":
        buy_position = min(change_position,int(cash/(1+0.00025)*this_window.AskPrice1.iloc[-1]))
        position += buy_position
        cash -= buy_position*this_window.AskPrice1.iloc[-1]*(1+0.00025)
        trans_record["time"].append(this_window.iloc[-1].name)  
        trans_record["tag"].append("long")
        trans_record["volume"].append(buy_position)
        trans_record["price"].append(np.float16(this_window.AskPrice1.iloc[-1]))
     elif tag == "short":
        sell_position = min(position,change_position)
        position -= sell_position
        cash += sell_position * this_window.AskPrice1.iloc[-1]*(1-0.00025)
        trans_record["time"].append(this_window.iloc[-1].name)
        trans_record["tag"].append("short")
        trans_record["volume"].append(sell_position)
        trans_record["price"].append(np.float16(this_window.AskPrice1.iloc[-1]))
    return position,cash,trans_record
    

def dtw(,ts2):
    
    m,n = len(ts1),len(ts2)

    D = np.zeros([n+1,m+1])

    D[:,0],D[0,:] = np.inf,np.inf

    D[0,0] = 0 
    
    for j in range(1,n+1):
        for i in range(1,m+1):
            D[j,i] = abs(ts1[i-1]-ts2[j-1]) + min(D[j-1,i-1],D[j-1,i],D[j,i-1]) 


    return D[n,m]

if __name__ == "__main__":

    import models
    ev = models.extreme_vision(20,300)
    df = ev._load_tick("000001","20170103","20170103")

    # do sampling, diminish the memory usage
    up_t = df.AskPrice1[200:600]
    down_t = df.AskPrice1[1700:2300]

    up_t = up_t.resample("30s").last()
    down_t = down_t.resample("30s").last()

    # transform original module ts into return curve
    ts_up = (up_t - up_t[0])/up_t[0]
    ts_down = (down_t - down_t[0])/down_t[0]
    backtest_df = ev._load_tick("000001","20170104","20170601")

    dates_list = backtest_df.index.normalize().unique().strftime("%Y%m%d")

    # set simulation transaction
    position = 0.0
    cash = 1000000.0
    fee = 0.00025
    equity_list = []
    trans_record = {"time":[],"tag":[],"volume":[],"price":[],}
    init_price = backtest_df.AskPrice1.iloc[0]
    end_price = backtest_df.AskPrice1.iloc[-1]
    change_position = 500


    window = 300
    td = pd.Timedelta(seconds=3)
    botm_list,peak_list = [0,],[0,]
    curve_ind_list = []

    for date_ in dates_list:
        print("now date:",date_)
        this_day = backtest_df.loc[date_]
        botm_diff,peak_diff = 0,0

        this_day_ind_ar = (np.linspace(1,4700,235)-1).astype(int) # minutes backtest

        for ind_ar in this_day_ind_ar:
            ind_ = this_day.index[ind_ar]
            curve_ind_list.append(ind_)
            equity_list.append(cash + position*this_day.loc[ind_].AskPrice1)
            try:
                this_window = this_day.loc[ind_-window*td:ind_]
                if len(this_window) < window:
                    continue
                ts0 = this_window.AskPrice1.resample("30s").last() # do sampling 30s
                if ts0.std() == 0:
                    continue
            except:
                continue

            # ts0 = (ts0 - ts0.mean())/ts0.std()
            ts0 = (ts0 - ts0[0])/ts0[0]
            botm_degree = dtw(ts0,ts_down)
            peak_degree = dtw(ts0,ts_up)

            if peak_degree < 0.140 and botm_degree> 0.036: # find peak
                botm_diff =  botm_degree - botm_list[-1]
                peak_diff = peak_degree - peak_list[-1]
                '''
                plt.plot(this_window.AskPrice1)
                plt.title("peak")
                plt.show()
                '''
                # sell if position > 0 and botm degree keep decreasing
                sell_position = min(position,change_position)
                if sell_position > 0 and botm_diff <= 0 and peak_diff >= 0:

            elif botm_degree < 0.005 and peak_degree > 0.190: # find bottom
                botm_diff = botm_degree - botm_list[-1]
                peak_diff = peak_degree - peak_list[-1]
                '''
                plt.plot(this_window.AskPrice1)
                plt.title("bottom")
                plt.show()
                '''
                buy_position = min(change_position,int(cash/((1+fee)*this_window.AskPrice1.iloc[-1])))


                # buy if cash is enough, and peak degree keep decreasing
                if buy_position > 0 and peak_diff <= 0 and botm_diff >= 0:
                    position += buy_position
                    cash -= buy_position * this_window.AskPrice1.iloc[-1]*(1+fee)
                    # pdb.set_trace()
                    trans_record["time"].append(this_window.iloc[-1].name)
                    trans_record["tag"].append("long")
                    trans_record["volume"].append(buy_position)
                    trans_record["price"].append(np.float16(this_window.AskPrice1.iloc[-1]))

            botm_list.append(botm_degree)
            peak_list.append(peak_degree)

            print(ind_)
            print("botm_degree: %s,peak_degree: %s"%(botm_degree,peak_degree))

    botm_ar = np.array(botm_list)
    peak_ar = np.array(peak_list)
    print("init_price:%s,end_price:%s"%(init_price,end_price))
    equity_curve = np.array(equity_list)
    equity_ret = (equity_curve - equity_curve[0])/equity_curve[0]

    buy_hold_ret = (backtest_df.AskPrice1 - backtest_df.AskPrice1[0])/backtest_df.AskPrice1[0]
    trans_record = pd.DataFrame(trans_record)

    plt.plot(equity_ret,label="investment")
    plt.plot(buy_hold_ret.loc[curve_ind_list].values,label="buy&hold")
    plt.legend()
    plt.show()
    pdb.set_trace()
