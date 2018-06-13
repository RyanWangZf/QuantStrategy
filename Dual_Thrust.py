
# -*- coding: utf-8 -*-


import numpy as np
import models
import pandas as pd
import pdb
import matplotlib.pyplot as plt


def operation(position,cash,trans_record,tag,price,time_,change_position=100):
    if tag == "long":
        buy_position = min(change_position,int(cash/((1+0.0003)*price)))
        position += buy_position
        cash -= buy_position*price*(1+0.0003)
        trans_record["time"].append(time_)
        trans_record["tag"].append("long")
        trans_record["volume"].append(buy_position)
        trans_record["price"].append(price)
    elif tag == "short":
        sell_position = min(position,change_position)
        position -= sell_position
        cash += sell_position * price *(1-0.0003)
        trans_record["time"].append(time_)
        trans_record["tag"].append("short")
        trans_record["volume"].append(sell_position)
        trans_record["price"].append(price)
    return position,cash,trans_record

def dual_thrust(df,df_back,k1=0.3,k2=0.3):
    hh = df_back.AskPrice1.max()
    ll = df_back.AskPrice1.min()
    close_ = []
    date_list = df_back.index.normalize().unique().strftime("%Y%m%d")
    for d_ in date_list:
        close_.append(df_back.loc[d_].AskPrice1[-1])
    lc = min(close_)
    hc = max(close_)
    range_ = max(hh-lc,hc-ll)

    up_bound = df.AskPrice1[0] + k1 * range_
    down_bound = df.AskPrice1[0] - k2 * range_
    return up_bound,down_bound

if __name__ == "__main__":

    dbloader = models.basemodel()
    start_date = "20170301"
    end_date = "20170401"

    backtest_df = dbloader._load_tick("000001",start_date,end_date)
    hs300 = dbloader._load_tick("999999",start_date,end_date)
    date_list = backtest_df.index.normalize().unique().strftime("%Y%m%d")

    position = 50000
    cash = 0.0
    fee = 0.0003
    equity_list = []
    curve_ind_list= []
    trans_record = {"time":[],"tag":[],"volume":[],"price":[],}
    init_price = backtest_df.AskPrice1.iloc[0]
    end_price = backtest_df.AskPrice1.iloc[-1]


    # start backtest

    for date_ in date_list:
        long_count = 0
        short_count = 0
        try:
            this_day = backtest_df.loc[date_]
        except:
            continue

        df_back = dbloader._load_tick_back(this_day,back_days=7)
        # get bounds of dual thrust
        up_bound,down_bound = dual_thrust(this_day,df_back,k1=0.3,k2=0.1)
        ''' 
        plt.plot([up_bound]*len(this_day),label="up_bound")
        plt.plot([down_bound]*len(this_day),label="down_bound")
        plt.plot(this_day.AskPrice1.values,label="price")
        plt.legend()
        plt.show()
        pdb.set_trace()
        '''
        ind_ar = this_day.resample("min").last().index

        for ind_ in ind_ar:
            try:
                equity_list.append(cash+position*this_day.loc[ind_].AskPrice1)
            except:
                continue
            curve_ind_list.append(ind_)
            print(ind_,equity_list[-1])
            askprice = float(this_day.loc[ind_].AskPrice1)

            if askprice > up_bound and short_count < 2: # short on cell
                position,cash,trans_record = operation(position,cash,trans_record,\
                    "short",askprice,ind_,10000)
                short_count += 1

            elif askprice < down_bound and long_count < 2: # long on bottom
                position,cash,trans_record = operation(position,cash,trans_record,\
                    "long",askprice,ind_,10000)
                long_count += 1

        # clean position at last in every day
        askprice = float(this_day.AskPrice1.iloc[-1])
        change_position = int(cash / ((1+fee)*askprice))
        position += change_position
        cash -=  askprice*(1+fee)*change_position


    # show the backtest results
    equity_curve = np.array(equity_list)
    equity_ret = (equity_curve - equity_curve[0])/equity_curve[0]
    buy_hold_ret = (backtest_df.AskPrice1 - backtest_df.AskPrice1[0])/backtest_df.AskPrice1[0]
    trans_record = pd.DataFrame(trans_record)
    trans_record = trans_record.loc[trans_record["volume"]>0].reset_index(drop=True)
    hs300_ret = (hs300.Price - hs300.Price[0])/hs300.Price[0]
    plt.plot(equity_ret,label="investment")
    plt.plot(buy_hold_ret.loc[curve_ind_list].values,label="buy&hold")
    plt.plot(hs300_ret.loc[curve_ind_list].values,label="index")
    plt.legend()
    plt.show()
    print("mean_buy_price:",trans_record.loc[trans_record["tag"]=="long"].price.mean())
    print("mean_short_price:",trans_record.loc[trans_record["tag"]=="short"].price.mean())

    pdb.set_trace()
    
   
