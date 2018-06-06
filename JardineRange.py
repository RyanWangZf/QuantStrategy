# -*- coding: utf-8 -*-
import models
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt


def operation(position,cash,trans_record,tag,price,time_,change_position=100):
    if tag == "long":
       buy_position = min(change_position,int(cash/((1+0.00025)*price)))
       position += buy_position
       cash -= buy_position*price*(1+0.00025)
       trans_record["time"].append(time_)
       trans_record["tag"].append("long")
       trans_record["volume"].append(buy_position)
       trans_record["price"].append(price)
    elif tag == "short":
       sell_position = min(position,change_position)
       position -= sell_position
       cash += sell_position * price *(1-0.00025)
       trans_record["time"].append(time_)
       trans_record["tag"].append("short")
       trans_record["volume"].append(sell_position)
       trans_record["price"].append(price)
    return position,cash,trans_record

if __name__ == "__main__":

    ev = models.trend_scout(10,100)
    start_date = "20170101"
    end_date = "20170301"
    backtest_df = ev._load_tick("000001",start_date,end_date)
    hs300 = ev._load_tick("999999",start_date,end_date)
    # set simulation transaction
    position = 0.0
    cash = 1000000.0
    fee = 0.00025
    equity_list,curve_ind_list = [],[]
    trans_record = {"time":[],"tag":[],"volume":[],"price":[],}
    init_price = backtest_df.AskPrice1.iloc[0]
    end_price = backtest_df.AskPrice1.iloc[-1]

    # get last 30 days vpc
    df_back = ev._load_tick_back(backtest_df,back_days=30)
    vpc = []
    back_date_list = df_back.index.normalize().unique().strftime("%Y%m%d")
    for date_ in back_date_list:
        vpc.append(df_back.loc[date_].AskPrice1.value_counts().argmax())
    vpc_ar = np.unique(vpc)

    # start simulation transaction
    date_list = backtest_df.index.normalize().unique().strftime("%Y%m%d")
    equity_list = []
    days_counter = 1
    for date_ in date_list:
        try: # try if this is open market day
            this_day = ev._load_tick("000001",date_,date_)
        except:
            continue
        if np.mod(days_counter,30) == 0 and days_counter > 1: # update vpc every month
            print("update vpc...")
            vpc = []
            df_back = ev._load_tick_back(this_day,30)
            back_date_list = df_back.index.normalize().unique().strftime("%Y%m%d")
            for d_ in back_date_list:
                vpc.append(df_back.loc[d_].AskPrice1.value_counts().argmax())
            vpc_ar = np.unique(vpc)
        ind_ar = (np.linspace(1,4700,235)-1).astype(int)
        open_price = this_day.AskPrice1[0]
        try:
            upper_vpc = np.float(np.sort(vpc_ar[vpc_ar>open_price])[0])
        except: # open price > all vpc
            upper_vpc = open_price + 0.01

        try:
            down_vpc = np.float(np.sort(vpc_ar[vpc_ar<open_price])[-1])
        except: # open price < all vpc
            down_vpc = open_price - 0.01

        upper_break = 0
        down_break = 0
        for ind_ in ind_ar:
            ind_ = this_day.index[ind_]
            equity_list.append(cash+position*this_day.loc[ind_].AskPrice1)
			curve_ind_list.append(ind_)
            print(ind_)
            askprice = np.float(this_day.loc[ind_].AskPrice1)
            if askprice <= down_vpc and down_break <=  0 : # long
                position,cash,trans_record = operation(position,cash,trans_record,\
                    "long",askprice,ind_,10000)
                down_break += 1

            elif askprice >= upper_vpc and upper_break <= 0 : # short
                position,cash,trans_record = operation(position,cash,trans_record,\
                    "short",askprice,ind_,10000)
                upper_break += 1

        days_counter += 1
        # update vpc every day
        new_vpc = this_day.AskPrice1.value_counts().argmax()
        peak_price = this_day.AskPrice1.max()
        botm_price = this_day.AskPrice1.min()
        vpc_ar = vpc_ar[(vpc_ar>=peak_price)|(vpc_ar<=botm_price)]
        vpc_ar = np.r_[vpc_ar,new_vpc]




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
