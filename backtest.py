# -*- coding: utf-8 -*-
# Python 3.5 


from aiye_data_loader_ticks2 import DBLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import logging
import gc

logging.basicConfig(level=logging.DEBUG)

class back_test():

    def __init__(self,sid,start_date,end_date,cash=10000000.0):
        # initialize global parameters
        self.start_date = start_date
        self.end_date = end_date
        self.cash = cash
        self.position = 0.0 
        self.look_back = 7 
        self.fee = 0.0025 # trade fee percent
        self.sid = sid 
        self.equity_curve = pd.Series()
    
    def _load_tick(self,this_date):
        # load ticks data (this day) fo doing buy and sell 
        # i.e. "20180509"
        # attention that some ticks' price is 0, which means suspending trade of the stock
        db = DBLoader()
        df = db.load_tick(sid=self.sid,start_date=this_date,end_date=this_date,\
                          start_time = 93000000,end_time=145500000)
        askprice1,bidprice1 = df.AskPrice1,df.BidPrice1
        date = askprice1.index.normalize().unique()
        if date.shape[0] > 1:
            raise ValueErorr("this date contains more than one day")
        date_ = date[0].strftime("%Y%m%d")
        ask1_ = askprice1.loc[date_].resample('1s',label='right',\
                              closed='right').last().fillna(method="ffill")
        ask1_ = pd.concat([ask1_.between_time("9:30:01","11:30"),\
                              ask1_.between_time("13:00:01","14:55")])
        bid1_ = bidprice1.loc[date_].resample('1s',label='right',\
                              closed='right').last().fillna(method="ffill")
        bid1_ = pd.concat([bid1_.between_time("9:30:01","11:30"),\
                           bid1_.between_time("13:00:01","14:55")])
        ask1_ = ask1_.loc[df.index].dropna()
        bid1_ = bid1_.loc[df.index].dropna()
	this_day_tick = {"askprice1":ask1_,"bidprice1":bid1_}
        this_day_tick = pd.DataFrame(this_day_tick,index=ask1_.index)
        logging.debug("%s contains %s ticks"%(date_,this_day_tick.shape[0]))
        self.this_day_tick = this_day_tick.copy()
        del this_day_tick;gc.collect()

    def _load_data(self,sid,start_date,end_date,start_time=93000000,end_time=145500000):
        # load ticks from start_date to end_date
        # every day ticks' length ought to be same as this_day_tick
        db = DBLoader()
        df = db.load_tick(sid = sid,start_date=start_date,\
                        end_date=end_date,start_time=start_time,\
                          end_time=end_time)
        askprice1,bidprice1 = df.AskPrice1,df.BidPrice1
        askprice1[askprice1 == 0] = np.nan
        bidprice1[bidprice1 == 0] = np.nan
        askprice1.fillna(method='pad',inplace=True)
        bidprice1.fillna(method='pad',inplace=True)
        midprice = (askprice1 + bidprice1)/2
        volume = df.Volume
        dates = midprice.index.normalize().unique()
        res,res_vol = pd.Series(),pd.Series()
        for date_ in dates:
            date_ = date_.strftime("%Y%m%d")
            mid_d = midprice.loc[date_].resample('1s',label='right',\
                        closed='right').last().fillna(method='ffill')
            mid_d = pd.concat([mid_d.between_time("9:30:01","11:30"),\
                        mid_d.between_time("13:00:01","14:55")])
            res = pd.concat([res,mid_d],axis=0)
            vol_d = volume.loc[date_].resample('1s',label='right',\
                                closed='right').last().fillna(method='ffill')
            vol_d = pd.concat([vol_d.between_time("9:30:01","11:30"),\
                                vol_d.between_time("13:00:01","14:55")])
            res_vol = pd.concat([res_vol,vol_d],axis=0)

        # set window(s) e.g. if window = 90s, the number of ticks is 90/3 = 30 ticks
        window = 90
        mid = res.sort_index()
        vol = res_vol.sort_index()
        y = (mid.shift(-window)-mid)/mid
        y = y.loc[df.index]
        vol = vol.loc[df.index]
        midprice = mid.loc[df.index]
        mid = mid.loc[df.index]
        midprice = midprice[mid.notnull()]
        ret = y[mid.notnull()]
        vol = vol[mid.notnull()]
        dataframe = pd.concat([vol,ret,midprice],axis=1)
        dataframe.columns = ['volume','ret','price']
        # attention: ret col has 30 NaN at its last 30 rows
        # dataframe = dataframe.dropna()
        return dataframe

    def _load_data_back(self,df,look_back):
        # load data looking back from df
        # for in-sample training
        dataset = df.copy()
        this_date = dataset.index[0]
        td1 = pd.Timedelta(days=look_back)
        td2 = pd.Timedelta(days=1)
        start_date = (this_date-td1).strftime("%Y%m%d")
        end_date = (this_date-td2).strftime("%Y%m%d")
        df_back = self._load_data(sid=self.sid,start_date=start_date,end_date=end_date)
        logging.debug("load back data: %s"%(df_back.shape[0]))
        del dataset; gc.collect()
        return df_back

    def _before_trading(self,this_date):
        # run once every day before open
        # get tag list of this day
        # i.e +n: long n, -n: short n, 0: no operation
        self._load_tick(this_date=this_date)
        # out-sample df
        df = self._load_data(sid=self.sid,start_date=this_date,end_date=this_date)
        # in-sample df
        df_back = self._load_data_back(df,look_back=self.look_back)
        # prediction of this day's trade signals
        self._calculate(df_back,df)

    def _handle_tick(self):
        # run on every tick, make decisions
        # this day's ticks saved in self.this_day_tick
        tag_list = self.tag_list.copy()
        this_day_tick = self.this_day_tick.copy()
        position = self.position
        cash = self.cash
        ask = this_day_tick["askprice1"]
        bid = this_day_tick["bidprice1"]

        position_flow = position + tag_list.cumsum()
        # attention when position < 0
        counter = 0
        t_delta = tag_list.index[1] - tag_list.index[0]
        pos_below_zero_ind = position_flow[position_flow<0].index
        while pos_below_zero_ind.shape[0] > 0:
            pos_last = position_flow[position_flow.index < pos_below_zero_ind[0]]
            if pos_last.shape[0] == 0:
                tag_list[tag_list.index[0]] = -1 * position
            else:
                tag_list[pos_below_zero_ind[0]] = -1 * pos_last.iloc[-1]

            position_flow = position + tag_list.cumsum()
            counter += 1
            pos_below_zero_ind = position_flow[position_flow<0].index

            if counter > len(tag_list):
                raise ValueError("unknown error happened: position is below zero")
        position_flow = position + tag_list.cumsum()

        long_ind = tag_list[tag_list>0].index
        short_ind = tag_list[tag_list<0].index
        keep_ind = tag_list[tag_list==0].index
        # ignore the circumstance when cash < 0
        # take trade fee into consideration
        cash_flow = pd.concat([-1*(1+self.fee)*tag_list[long_ind]*ask[long_ind],\
                               -1*(1-self.fee)*tag_list[short_ind]*bid[short_ind],\
                                tag_list[keep_ind]*0.0],axis=0).sort_index()
        cash_cumsum = cash + cash_flow.cumsum()
        this_equity_curve = position_flow * ask + cash_cumsum
        self.equity_curve = pd.concat([self.equity_curve,this_equity_curve],axis=0).sort_index()
        # print("this_equity_curve",this_equity_curve[-10:])
        del this_equity_curve;gc.collect()
        # save today's condition
        self.position = position_flow.iloc[-1]
        self.cash = cash_cumsum.iloc[-1]
        self.equity = self.equity_curve.iloc[-1]
        logging.debug("handle tick compeleted, position: %s"%(self.position))
        logging.debug("handle tick compeleted, cash: %s"%(self.cash))
        logging.debug("handle tick compeleted, equity: %s"%(self.equity))

    def _after_trading(self):
        # run once every day after close
        pass

    def _calculate(self,insample,outsample):
        # doing calculations, get tags of every tick
        # output pd.Seires()
        df_back = insample.copy()
        df = outsample.copy()
        tag_list = np.zeros(len(df))
        tag_list[0] = 100000
        tag_list[-1] = -100000
        tag_list = pd.Series(tag_list,name="tags",index=df.index)
        self.tag_list = tag_list.copy()
        del tag_list;gc.collect()
        logging.debug("calculate completed, tag_list length: %s"%(self.tag_list.shape[0]))

    def run(self):
        # start running the back test
        df_samples = self._load_data(self.sid,self.start_date,self.end_date)
        self.date_list = [date.strftime("%Y%m%d") for date in \
                            df_samples.index.normalize().unique()]

        for date in self.date_list:
            self._before_trading(this_date=date)
            self._handle_tick()
            self._after_trading()

        logging.debug("backtest completed")
        logging.debug("equity: %s"%(self.equity))
        # after simulation of auto-trade completed    
        # call plot_curve() get return curve 
        # call benchmarks() get benchmarks of return etc.





