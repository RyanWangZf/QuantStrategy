# -*- coding: utf-8 -*-
from aiye_data_loader_ticks2 import DBLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import logging
import gc
import pdb 
    
logging.basicConfig(level=logging.DEBUG)
np.random.seed(84) # set random seed


class backtest():

    def __init__(self,sid,start_date,end_date,cash=10000000.0):
        # initialize global parameters
        np.random.seed(84)
        self.start_date = start_date
        self.end_date = end_date
        self.cash = cash
        self.equity = cash
        self.position = 0.0 
        self.look_back = 7 
        self.fee = 0.0025 # trade fee percent
        self.sid = sid 
        self.equity_curve = pd.Series()
        logging.debug("back test start from %s to %s"%(start_date,end_date))
    
    def _load_tick(self,sid,start_date,end_date,start_time=93000000\
                            ,end_time=145500000):
        # basic loader: return cleaned raw tick data
        # attention: this does not process 0 samples in column Price
        db = DBLoader()
        df = db.load_tick(sid=sid,start_date=start_date,end_date=end_date,start_time=start_time,\
                    end_time=end_time)
        dates = df.index.normalize().unique()
        res = pd.DataFrame()
        for date in dates:
            date_ = date.strftime("%Y%m%d")
            df1_ = df.loc[date_].resample("1s",label="right",closed="right").last().fillna(method="ffill")
            df1_ = pd.concat([df1_.between_time("9:30:01","11:30"),\
                            df1_.between_time("13:00:01","14:55")],axis=0)
    
            res = pd.concat([res,df1_],axis=0)
        res = res.resample("3s",label="right",closed="right").last().fillna(method="ffill")
        res = pd.concat([res.between_time("9:30:01","11:30"),res.between_time("13:00:01","14:55")])
        # res = res.loc[df.index]
        # res = res.loc[res.Value.notnull()] # drop NaN
        col_sid = pd.Series([sid]*len(res),name="sid",index=res.index)
        res = pd.concat([res,col_sid],axis=1)
        logging.debug("load tick %s: %s"%(sid,res.shape[0]))
        return res

    def _load_tick_back(self,df,back_days=0):
        # input dataframe, out look back dataframe
        dataset = df.copy()
        this_date = dataset.index[0]
        td1 = pd.Timedelta(days=back_days)
        td2 = pd.Timedelta(days=1)
        sid = df.sid[0]
        start_date = (this_date - td1).strftime("%Y%m%d")
        end_date = (this_date - td2).strftime("%Y%m%d")
        df_back = self._load_tick(sid=sid,start_date=start_date,end_date=end_date)
        logging.debug("load tick back: %s"%(df_back.shape[0]))
        del dataset;gc.collect()
        return df_back

    def _train_model(self,df):
        # run every day before trading
        # return class model with method: train() & predict() & fit()
        from models import extreme_vision
        ev = extreme_vision(pos_backdays=30,window=100)
        self.backward_window = 100
        self.forward_window = 50
        ev.train(df,mode="bottom")
        self.predictor = ev


    def _before_trading(self,this_date):
        # run once every day before open
        # get tag list of this day
        # i.e +n: long n, -n: short n, 0: no operation
        self.this_day_tick = self._load_tick(self.sid,this_date,this_date)

    def _handle_tick(self):
        # run on every tick, make decisions

        # this day's ticks saved in self.this_day_tick
        try:
            predictor = self.predictor
        except:
            raise Exception("cannot load pre-trained model")
        this_day_tick = self.this_day_tick.copy()
        this_date = this_day_tick.index[0].strftime("%Y-%m-%d")
        logging.debug("do backtest on the ticks in within %s"%this_date)
        position = self.position
        cash = self.cash
        equity_list = []
        td = pd.Timedelta(seconds=3)
        start_tick = this_day_tick.index[0] + (self.backward_window+1)*td
        end_tick = this_day_tick.index[-1] - (self.forward_window) * td
        ind_list = this_day_tick.index
        # ind_list = [ind for ind in ind_list if ind >= start_tick and ind <= end_tick]
        proba_list = []
        for ind in ind_list:
            pred_proba = predictor.predict_on_tick(this_day_tick,ind)

            proba_list.append(pred_proba)

            bidprice = this_day_tick.loc[ind].BidPrice1
            equity_list.append(bidprice*position+cash)
            if pred_proba == 1.0 : # change position
                askprice = this_day_tick.loc[ind].AskPrice1
                buy_position = 10000
                if cash - buy_position * askprice < 0 : # cash is exhausted
                    buy_position = int(cash / askprice)
                position = position + buy_position
                cash = cash - (1+self.fee) * buy_position * askprice
        equity_list = pd.Series(equity_list,index=ind_list)
        self.equity_curve = pd.concat([self.equity_curve,equity_list],\
                                        axis=0) # record today equity change
        pdb.set_trace()
        self.position = position
        self.cash = cash
        self.equity = equity_list.values[-1]
        logging.debug("handle tick compeleted, position: %s"%(self.position))
        logging.debug("handle tick compeleted, cash: %s"%(self.cash))
        logging.debug("handle tick compeleted, equity: %s"%(self.equity))


    def _handle_tick_quick(self):
	
        # this version receive tag_list from _calculate every day instead of loop in ind_list
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
        bidprice = self.this_day_tick.BidPrice1.iloc[-1]
        self.cash = self.position * bidprice * (1-self.fee) + self.cash
        self.position = 0.0
        # try doing incremental learning every day after trading
        # self.predictor.update(self.this_day_tick,mode="bottom")


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
        df_samples = self._load_tick(self.sid,self.start_date,self.end_date)
        self.date_list = [date.strftime("%Y%m%d") for date in \
                            df_samples.index.normalize().unique()]

        # training model
        self._train_model(df_samples)

        for date in self.date_list:
            self._before_trading(this_date=date)
            self._handle_tick()
            self._after_trading()

        logging.debug("backtest completed")
        logging.debug("equity: %s"%(self.equity))
        # after simulation of auto-trade completed    
        # call plot_curve() get return curve 
        # call benchmarks() get benchmarks of return etc. 

    def plot_curve():
        import matplotlib.pyplot as plt
        if len(self.equity_curve) < 1:
            raise Exception("call plot curve before running backtest")
        plt.plot(self.equity_curve)
        plt.title("performance of equity curve")
        plt.show()

    def benchmark():
        pass













