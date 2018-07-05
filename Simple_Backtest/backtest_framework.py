# -*- coding: utf-8 -*-
# Python 3.5 on Linux
# 

from aiye_data_loader_ticks2 import DBLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import gc
import pdb

logging.basicConfig(level=logging.DEBUG)
np.random.seed(84) # set random seed
"the server has maximum 40 threads"

class backtest():

    def __init__(self,sid,start_date,end_date,cash=10000000.0):
        # initialize global parameters
        np.random.seed(84)
        self.start_date = start_date
        self.end_date = end_date
        self.cash = cash
        self.equity = cash
        self.position = 0.0
        self.fee = 0.00025 # trade fee percent
        self.sid = sid
        self.frozen_position =  0.0 # restriction of T+1 sell
        self.equity_curve = pd.Series()
        self.trans_record = {"time":[],"tag":[],"volume":[],"price":[]}
        logging.debug("back test start from %s to %s"%(start_date,end_date))

    def _load_tick(self,sid,start_date,end_date,start_time=93000000\
                            ,end_time=145500000):
        # basic loader: return cleaned raw tick data
        # attention: this does not process 0 samples in column Price
        db = DBLoader()
        if sid not in ["999999"]:
            df = db.load_tick(sid=sid,start_date=start_date,end_date=end_date,start_time=start_time,\
                    end_time=end_time)
        else:
            df = db.load_index(sid=sid,start_date=start_date,end_date=end_date,start_time=start_time,\
                    end_time=150000000)
        dates = df.index.normalize().unique()
        res = pd.DataFrame()
        for date in dates:
            date_ = date.strftime("%Y%m%d")

            df1_ = df.loc[date_].resample("1s",label="right",\
                          closed="right").last().fillna(method="ffill")
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
        sid = df.sid[0]
        tag = True
        while tag:
            td1 = pd.Timedelta(days=back_days)
            td2 = pd.Timedelta(days=1)
            try:
                start_date = (this_date - td1).strftime("%Y%m%d")
                end_date = (this_date - td2).strftime("%Y%m%d")
                df_back = self._load_tick(sid=sid,start_date=start_date,end_date=end_date)
                logging.debug("load tick back: %s"%(df_back.shape[0]))
                tag = False
            except:
                back_days += 1
            if back_days > 10000:
                raise Exception("Unknown error happened, Timed out.")
        del dataset;gc.collect()
        return df_back

    def _train_model(self,df):
        # run every day before trading
        # return class model with method: train() & predict() & fit()
        from models import trend_scout
        sc = trend_scout(pos_backdays=30,window =300)
        self.backward_window = 300
        self.forward_window = 50
        sc.train(df)
        print("model extreme vision trained.")
        self.predictor = sc

    def _trade_operation(self,position,cash,tag,time_,change_position=100,price=0):

        if tag == "long":
            buy_position = min(change_position,int(cash/((1+0.00025)*price)))
            position += buy_position
            cash -= buy_position*price*(1+0.00025)
            self.trans_record["time"].append(time_)
            self.trans_record["tag"].append(tag)
            self.trans_record["volume"].append(buy_position)
            self.trans_record["price"].append(np.float16(price))

        elif tag == "short":
            sell_position = min(position,change_position)
            position -= sell_position
            cash += sell_position * price *(1-0.00025)
            self.trans_record["time"].append(time_)
            self.trans_record["tag"].append(tag)
            self.trans_record["volume"].append(sell_position)
            self.trans_record["price"].append(np.float16(price))

        return position,cash

    def _before_trading(self,this_date):
        # run once every day before open
        # get tag list of this day
        # i.e +n: long n, -n: short n, 0: no operation
        self.this_day_tick = self._load_tick(self.sid,this_date,this_date)

    def _handle_tick(self):
        try:
            predictor = self.predictor
        except:
            raise Exception("cannot load pretrained model.")
        this_day_tick= self.this_day_tick.copy()
        this_day_feat,feature_list = predictor.generate_feature(this_day_tick)
        this_day_feat = this_day_feat.dropna()

        position = self.position
        cash = self.cash
        equity_list = []
        curve_ind_list = []
        td = pd.Timedelta(seconds=3)

        # the following is trade logic
        this_ind_ar = (np.linspace(1,4700,235)-1).astype(int) # on evert minute
        proba_list = []
        for ind_ar in this_ind_ar:
            ind_ = this_day_tick.index[ind_ar]
            print(ind_)
            curve_ind_list.append(ind_)
            equity_list.append(cash+position*this_day_tick.loc[ind_].AskPrice1)
            pred_proba = predictor.predict_on_tick(this_day_feat,ind_)
            proba_list.append(pred_proba)
            if pred_proba[1] > 0.9 and pred_proba[2] < 0.5: # find botm
                position,cash = self._trade_operation(position,cash,"long",ind_,change_position=1000,\
                        price = this_day_tick.loc[ind_].AskPrice1)

            elif pred_proba[2] > 0.9 and pred_proba[1] < 0.5: # find peak
                position,cash = self._trade_operation(position,cash,"short",ind_,change_position=1000,\
                        price = this_day_tick.loc[ind_].AskPrice1)



        equity_list = pd.Series(equity_list,index=curve_ind_list)
        self.equity_curve = pd.concat([self.equity_curve,equity_list],\
                                        axis=0) # record today equity change
        "params for test"
        ar_proba= np.array(proba_list)
        botm = ar_proba[:,1]
        peak = ar_proba[:,2]
        ask_price = this_day_tick.AskPrice1
        # plt.plot(this_day_tick.loc[this_day_tick.index[this_ind_ar]].AskPrice1.values)
        self.position = position
        self.cash = cash
        self.equity = equity_list.values[-1]

    def _after_trading(self):
        # run once every day after close


        self.frozen_position = 0.0 # reset frozen position
        askprice = self.this_day_tick.AskPrice1.iloc[-1]
        self.cash = self.position * askprice * (1-self.fee) + self.cash
        self.position = 0.0
        self.equity = self.cash
        logging.info("handle tick compeleted, position: %s"%(self.position))
        logging.info("handle tick compeleted, cash: %s"%(self.cash))
        logging.info("handle tick compeleted, equity: %s"%(self.equity))

        # try doing incremental learning every day after trading
        # self.predictor.update(self.this_day_tick,mode="bottom")

    def run(self):
        # start running the back test
        df_samples = self._load_tick(self.sid,self.start_date,self.end_date)
        self.date_list = [date.strftime("%Y%m%d") for date in \
                            df_samples.index.normalize().unique()]

        # training model
        self._train_model(df_samples)

        for date in self.date_list:
            try:
                self._before_trading(this_date=date)
            except:
                logging.debug("%s is not trading day, skip it"%(date))
                continue
            self._handle_tick()
            self._after_trading()

        logging.debug("backtest completed")
        logging.debug("equity: %s"%(self.equity))
        df_trans= pd.DataFrame(self.trans_record)
        self.trans_record = df_trans.loc[df_trans.volume>0].sort_values(by=\
                                  "time").reset_index(drop=True)
        # after simulation of auto-trade completed    
        # call plot_curve() get return curve 
        # call benchmarks() get benchmarks of return etc. 

    def plot_curve(self):
        import matplotlib.pyplot as plt
        if len(self.equity_curve) < 1:
            raise Exception("call plot curve before running backtest")
        plt.plot(self.equity_curve)
        plt.title("performance of equity curve")
        plt.show()


    def benchmark(self):
        try:
            equity_curve = self.equity_curve.copy()
        except:
            raise Exception("call equity_curve before running backtest.")

        ret_curve = (equity_curve - equity_curve.iloc[0])/self.equity_curve.iloc[0]

        hs300 = self._load_tick(sid="999999",start_date=self.start_date,\
                                end_date=self.end_date)

        hs_curve = (hs300.Price - hs300.Price.iloc[0])/hs300.Price.iloc[0]

        ret_curve = ret_curve.resample("1d").last().fillna(method="ffill")
        hs_curve = hs_curve.resample("1d").last().fillna(method="ffill")
        ret_curve = ret_curve.loc[hs_curve.index] # keep same length
        ret_curve.fillna(method="ffill",inplace=True)
        # alpha
        alpha_ = ret_curve.iloc[-1] - hs_curve.iloc[-1]
        # beta
        beta_ = np.polyfit(hs_curve,ret_curve,1)[0]
        # sharpe ratio
        sharpe_ratio = ret_curve.iloc[-1]/ret_curve.std()
        # alpha curve
        alpha_curve = ret_curve - hs_curve
        alpha_curve.fillna(method="ffill",inplace=True)

        print("Alpha:",alpha_)
        print("Beta:",beta_)
        print("Sharpe Ratio:",sharpe_ratio)

        # average volume-weighted short price & long price
        trans_df = self.trans_record
        vwap_short = trans_df.loc[trans_df.tag=="short"]
        vwap_short = (vwap_short.price*vwap_short.volume).sum()/vwap_short.volume.sum()

        vwap_long = trans_df.loc[trans_df.tag=="long"]
        vwap_long = (vwap_long.price*vwap_long.volume).sum()/vwap_long.volume.sum()
        print("vol-weighted average buy price :",vwap_long)
        print("vol-weighted average sell price :",vwap_short)

        plt.subplot(211)
        plt.plot(alpha_curve.values,c="red",alpha=0.75)
        plt.title("alpha curve of investment")
        plt.subplot(212)

        plt.plot(hs_curve.values,c="orange",alpha=0.75,label="index")
        plt.plot(ret_curve.values,c="red",label="investment")
        plt.title("return curve of market index and investment")
        plt.legend()
        plt.show()



if __name__ == "__main__":

    bt = backtest("000001","20170101","20170201")
    bt.run()

    pdb.set_trace()
    print("backtest completed.")


