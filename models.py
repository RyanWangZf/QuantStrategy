# -*- coding: utf-8 -*-
# Python 3.5 on Linux
# author : Ryan Wang
# 2018/5

import numpy as np
import pandas as pd
import logging
import pdb 
import gc
from aiye_data_loader_ticks2 import DBLoader

logging.basicConfig(level=logging.DEBUG)
np.random.seed(84) #set random seed

def _timer(func):
    from functools import wraps
    import time
    @wraps(func)
    def wrapper(*args,**kwargs):
        st =  time.time()
        result = func(*args,**kwargs)
        time_used = str(int(time.time()-st))
        print("%s finished, use %s s."%(func.__name__,time_used))
        return func(*args,**kwargs)
    return wrapper

class basemodel():
    # basemodel contains several base methods what a model may require
    def __init__(self):
        pass

    def _feature_select(self,x,k=5,target_name="ret"):
        # input features, select TOP k ones
        from sklearn.feature_selection import mutual_info_regression
        scorer = mutual_info_regression
        df_feature = x.loc[x[target_name].notnull()].drop([target_name],axis=1)
        target = x[target_name].dropna()
        try:
            mi = scorer(df_feature,target)
        except:

            mi = scorer(df_feature.values.reshape(-1,1),target)
        selector = pd.DataFrame({"mi":mi,"feature":df_feature.columns.values})
        selected_features = selector.sort_values(by="mi",ascending=False).feature[:k].values.tolist()
        logging.debug("feature select: %s"%selected_features)
        return selected_features

    def _load_tick(self,sid,start_date,end_date,start_time=93000000,end_time=145500000):
        db = DBLoader()
        df = db.load_tick(sid=sid,start_date=start_date,end_date=end_date,\
                    start_time=start_time,end_time=end_time)
        dates = df.index.normalize().unique()
        res = pd.DataFrame()
        for date in dates:
            date_ = date.strftime("%Y%m%d")
            df1_ = df.loc[date_].resample("1s",label="right",closed="right").last().fillna(\
                                    method="ffill")
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
    
    def _load_bar(self,sid,start_date,end_date):
        df = pd.DataFrame()
        start_date = datetime.datetime.strptime(start_date,"%Y%m%d")
        end_date = datetime.datetime.strptime(end_date,"%Y%m%d")
        date_list = []
        while start_date <= end_date:
            date_list.append(start_date.strftime("%Y%m%d"))
            start_date += datetime.timedelta(days=1)
        close_,high_,low_,poc_,vol_ = [],[],[],[],[]
        date_index = []
        for date_ in date_list:
            try:
                this_day = self._load_tick(sid,date_,date_)
            except:
                continue
            # ffill 0 
            this_day.AskPrice1 = this_day.AskPrice1.replace(0,np.nan).fillna(method="ffill").values

            date_index.append(date_)
            high = this_day.AskPrice1.max()
            low = this_day.loc[this_day.AskPrice1>0].AskPrice1.min()
            poc_.append(this_day.AskPrice1.value_counts().argmax())
            high_.append(high)
            low_.append(low)
            close_.append(this_day.AskPrice1.iloc[-1])
            vol_.append(this_day.loc[this_day.AccVolume>0].AccVolume.iloc[-1])
        df["high"] = high_
        df["low"] = low_
        df["close"] = close_
        df["poc"] = poc_
        df["volume"] = vol_
        df.index = date_index
        return df

class extreme_vision(basemodel):

    def __init__(self,pos_backdays,window):
        self.neg_backdays = 20
        self.pos_backdays = pos_backdays
        self.window = window
        self.forward_window = int(window/2)
        self.backward_window = window
        np.random.seed(84)

    def _create_DNN(self):
        # return a DNN model
        from keras.models import Model
        from keras.layers import Input,Dense
        from keras.utils.np_utils import to_categorical
        try:
            inputs = Input(shape=(self.dnn_input_shape,))
        except:
            raise ValueError("dnn_input_shape undefined")
        dense1 = Dense(16,activation="relu")(inputs)
        dense2 = Dense(64,activation="relu")(dense1)
        dense3 = Dense(16,activation="relu")(dense2)
        # softmax layers: output 2 classes probability
        predictions = Dense(2,activation="softmax")(dense3)

        model  = Model(inputs=inputs,outputs=predictions)
        model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["acc"])
        logging.debug("DNN model compiled completed")

        return model

    def _create_CNN(self):
        from keras.models import Model
        from keras.layers import Input,Dense,Dropout,Flatten
        from keras.layers import Conv2D,MaxPooling2D
        from keras.utils.np_utils import to_categorical

        try:# col is num of features, row is size of window detected
            inputs = Input(shape=(self.cnn_rows,self.cnn_cols,1))
        except:
            raise Exception("cnn input shape is not defined")
        # all the Conv2D layers share data format : channel_last 
        cnn1 = Conv2D(32,(3,3),activation="relu",padding="same")(inputs)
        cnn2 = Conv2D(32,(3,3),activation="relu",padding="same")(cnn1)
        max_p1 = MaxPooling2D(pool_size=(2,2))(cnn2)
        drop1 = Dropout(0.25)(max_p1)

        cnn3 = Conv2D(64,(3,3),activation="relu",padding="same")(drop1)
        cnn4 = Conv2D(64,(3,3),activation="relu",padding="same")(cnn3)
        max_p2 = MaxPooling2D(pool_size=(2,2))(cnn4)
        drop2 = Dropout(0.25)(max_p2)

        flat = Flatten()(drop2)
        dense1 = Dense(256,activation="relu")(flat)
        drop3 = Dropout(0.5)(dense1)
        predictions = Dense(self.num_classes,activation="softmax")(drop3)

        model = Model(inputs=inputs,outputs=predictions)
        model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["acc"])
        logging.debug("DNN model compiled completed")
        return model


    def _generate_feature(self,df):
        dataset = df.copy()
        # midprice
        mid = 0.5 * (dataset.AskPrice1 + dataset.BidPrice1)
        dataset["mid"] = mid
        # moneyflow
        vol = dataset.Volume
        mf_diff = mid.diff().fillna(0)
        mf = vol*mf_diff
        dataset["moneyflow"] = mf.rolling(100).sum() # have NaN
        # pricerange100
        price_range = mid.rolling(100).max() - mid.rolling(100).min()
        dataset["price_range_100"] = price_range # have NaN
        # difference between askvolume1 and bidvolume1
        diff_ask_bid = dataset.AskVolume1 - dataset.BidVolume1
        dataset["ask_bid_vol_diff"] = diff_ask_bid

        feature_list = ["Volume","mid","moneyflow","price_range_100","ask_bid_vol_diff"]
        dataset = dataset[feature_list]
        return dataset,feature_list
    
    def _search_pos_ind(self,df_back,mode="bottom"):
        # return positive samples' index
        dataset = df_back.copy()
        date_list = [date.strftime("%Y%m%d") for date in dataset.index.normalize().unique().tolist()]
        pos_ind_list = []
        for date_ in date_list:
            this_ = dataset.loc[date_]
            ask1,bid1 = this_.AskPrice1,this_.BidPrice1
            mid = (ask1 + bid1) / 2
            td = pd.Timedelta(seconds=3)
            if mode == "bottom":
                pos_ind = mid.loc[mid == mid.min()].index
            elif mode == "peak":
                pos_ind = mid.loc[mid == mid.max()].index
            for ind_ in pos_ind:
                # judge if a sample has forward and backward window samples or not
                ind_f = ind_ + self.forward_window * td
                ind_b = ind_ - (self.backward_window+1) * td
                if ind_f in this_.index and ind_b in this_.index:
                    ts = this_.loc[ind_b:ind_f].copy()

                    unique_list = [ts[ft].unique().shape[0] for ft\
                                in ["Volume","Price","AskVolume1","BidVolume1"]]

                    if max(unique_list) > 1: # not all cols are same
                        pos_ind_list.append(ind_)

        del dataset,ts; gc.collect()
        # do sampling
        sampling = np.random.rand(len(pos_ind_list))
        pos_ind_list = [pos_ind_list[i] for i in range(len(sampling)) if sampling[i] <= 1.0]
        logging.debug("search positive index : %s"%(len(pos_ind_list)))
        return pos_ind_list

    def _search_neg_ind(self,df_back,pos_ind_list):
        # return negative samples' index
        # requires input pos_ind_list to drop positive samples's index
        dataset = df_back.copy()
        date_list = [date.strftime("%Y%m%d") for date in dataset.index.normalize().unique().tolist()]
        neg_ind_list = []
        td = pd.Timedelta(seconds=3)
        for date_ in date_list:
            this_ = dataset.loc[date_]
            neg_ind = this_.index
            new_neg_ind = []
            for ind_ in neg_ind.tolist():
                ind_b = ind_ - self.backward_window * td
                ind_f = ind_ + self.forward_window * td
                if ind_b in neg_ind and ind_f in neg_ind and ind_ not in pos_ind_list:
                    ts = this_.loc[ind_b:ind_f].copy()
                    unique_list = [ts[ft].unique().shape[0] for ft \
                                   in ["Volume","Price","AskVolume1","BidVolume1"]]
                    if max(unique_list) > 1:
                        new_neg_ind.append(ind_)

            # do sampling
            sampling = np.random.rand(len(new_neg_ind))
            neg_ind = [new_neg_ind[i] for i in range(len(new_neg_ind)) if sampling[i] <= 0.1]
            neg_ind_list.extend(neg_ind)

        logging.debug("search negative index : %s"%(len(neg_ind_list)))
        return neg_ind_list

    def _ts_generator_DNN(self,df,ind_df): # for DNN
        # generate samples as input of DNN model
        # attention this return contains only features without labels
        # generate one sample once, use DNN.fit on batch
        ind_list = ind_df["ind"]
        dataset = df.copy()
        feature_list = dataset.columns.values.tolist()
        td = pd.Timedelta(seconds = 3)
        for ind_ in ind_list:
            ft_list = pd.DataFrame()
            ind_f = ind_ + self.forward_window * td
            ind_b = ind_ - self.backward_window * td
            # better change it as dataset.loc[ind_b:ind_].copy()
            ts = dataset.loc[ind_b:ind_f].copy()
            '''
            unique_num = [ts[ft].unique().shape[0] for ft in feature_list]
            if 1 in unique_num:
                logging.debug("find all same column, skip it")
                continue
            '''
            for ft in feature_list:
                k_ts = (ts[ft] - ts[ft].iloc[0])[1:] / np.arange(1,len(ts))
                point_time = ind_.strftime("%Y%m%d_%H:%M:%S")
                ft_list["%s_k_%s"%(ft,point_time)] = k_ts

            #normalization
            feat_vec = ((ft_list - ft_list.min())/(ft_list.max()-ft_list.min())).values.flatten()
            yield feat_vec,ind_df["target"].loc[ind_df["ind"] == ind_].values[0]

    def _ts_generator_CNN(self,df,ind_df): # for CNN
        ind_list = ind_df["ind"]
        dataset = df.copy()
        feature_list = dataset.columns.values.tolist()
        td = pd.Timedelta(seconds=3)
        for ind_ in  ind_list:
            ft_space = pd.DataFrame()
            ind_b = ind_ - self.backward_window * td
            ts = dataset.loc[ind_b:ind_].copy()
            '''
            unique_list = [ts[ft].unique().shape[0] for ft in feature_list]
            if 1 in unique_list:
                logging.debug("find all same columns, skip it")
                continue
            '''
            for ft in feature_list:
                k_ts = (ts[ft] - ts[ft].iloc[0])[1:] / np.arange(1,len(ts))
                point_time = ind_.strftime("%Y%m%d_%H:%M:%S")
                ft_space["%s_k_%s"%(ft,point_time)] = k_ts
            # [0,1] normalization, fillna with 0.0 (NaN generated by all same cols)
            feat_vec = ((ft_space - ft_space.min())/(ft_space.max()-ft_space.min())).fillna(0.0).values
            label = ind_df["target"].loc[ind_df["ind"]==ind_].values[0]
            yield feat_vec,label

    def _fit_on_batch(self,generator,batch_size):
        from keras.utils.np_utils import to_categorical
        logging.debug("train on mini batch, batch size is %s"%batch_size)
        num_t,counter = 0,0
        model = self._create_DNN()
        while True:
            try:
                 feat_vec,label = next(generator)
                 if counter  == 0:
                    label_vec = to_categorical(label,num_classes=self.num_classes)
                    batch_feat = feat_vec
                    counter += 1
                    continue
                 elif counter / batch_size >= 1:
                    num_t += 1
                    counter = 0
                    model.train_on_batch(batch_feat,label_vec,class_weight={1.0:0.5,0.0:0.5})
                    logging.debug("training on batch %s"%num_t)
                 else:
                    label_vec = np.vstack([label_vec,to_categorical(label,num_classes=self.num_classes)])
                    batch_feat = np.vstack([batch_feat,feat_vec])
                    counter += 1
            except:
                 model.train_on_batch(batch_feat,label_vec,class_weight={1.0:0.5,0.0:0.5})
                 num_t += 1
                 logging.debug("loop complete, sum up %s loop"%num_t)
                 break

        return model,num_t

    @_timer
    def train(self,df,mode="bottom"):
        import os
        from keras.utils.np_utils import to_categorical
        if mode in ["bottom","peak"]:
            self.num_classes = 2

        file_list = os.listdir()
        model_name="CNN_model.h5"
        if model_name in file_list:
            from keras.models import load_model
            logging.info("find existed model %s, load it"%model_name)
            self.model = load_model(model_name)
        else:
            dataset = df.copy()
            df_back = self._load_tick_back(dataset,back_days=self.pos_backdays)
            pos_ind_list = self._search_pos_ind(df_back,mode=mode)
            pos_ind_list = pd.DataFrame(np.c_[pos_ind_list,\
                [1]*len(pos_ind_list)],columns=["ind","target"])
            df_back_neg = self._load_tick_back(df,back_days = self.neg_backdays)
            neg_ind_list = self._search_neg_ind(df_back_neg,pos_ind_list)
            neg_ind_list = pd.DataFrame(np.c_[neg_ind_list,\
                 [0]*len(neg_ind_list)],columns=["ind","target"])
            ind_list = pd.concat([pos_ind_list,neg_ind_list],axis=0)
            logging.debug("index list loaded, total get %s ticks"%(len(ind_list)))
            print("get ind from ",ind_list["ind"].iloc[0]," to ",ind_list["ind"].iloc[-1])
            ind_df = ind_list.sample(frac=1).reset_index(drop=True)
            back_days= max(self.pos_backdays,self.neg_backdays)
            df_back = self._load_tick_back(dataset,back_days = back_days)
            dataset,feature_list = self._generate_feature(df_back)
            dataset = dataset.dropna() # drop NaN
            self.cnn_cols = len(feature_list)
            self.cnn_rows = self.window
            # get array of feat_vec & label
            logging.debug("loading data from generator...")
            g = self._ts_generator_CNN(dataset,ind_df)
            tag = 0
            while True:
                try:
                    feat_vec,label = next(g)
                    if tag == 0:
                        x,y = feat_vec,label
                        x = x.reshape(1,x.shape[0],x.shape[1],1)
                        tag = 1
                    else:
                        x = np.r_[x,feat_vec.reshape(1,feat_vec.shape[0],feat_vec.shape[1],1)]
                        y = np.r_[y,label]
                except:
                    break
            model = self._create_CNN()
            len_train = int(0.9*len(y))
            x_train,y_train = x[:len_train],y[:len_train]
            x_test,y_test = x[len_train:],y[len_train:]
            y_train = to_categorical(y_train,num_classes=self.num_classes)
            y_test = to_categorical(y_test,num_classes=self.num_classes)
            "set the total epochs of training, got by observation"
            epochs = 45
            model.fit(x_train,y_train,batch_size=128,epochs=epochs,verbose=1,\
                    validation_data=(x_test,y_test))
            score = model.evaluate(x_test,y_test,verbose=0)
            logging.debug("model training complete")
            print("Test loss:",score[0])
            print("Test accuracy:",score[1])
            try:
                model.save("CNN_model.h5")
            except:
                logging.warning("fail saving model, check if module h5py exists")

            self.model = model
            del dataset,df_back,df_back_neg; gc.collect()

    @_timer
    def update(self,df,mode="bottom"):
        # do incremental learning upon new raw samples,(every day)
        import time
        try:
            model = self.model
        except:
            raise Exception("model has to be trained before doing incremental learning")
        dataset = df.copy()
        dataset,feature_list = self._generate_feature(dataset)
        dataset = dataset.dropna() #drop NaN

        pos_ind_list = self._search_pos_ind(dataset,mode=mode)
        pos_ind_list = pd.DataFrame(np.c_[pos_ind_list,[1]*len(pos_ind_list)],columns=["ind","target"])

        neg_ind_list = self._search_neg_ind(dataset,pos_ind_list)
        neg_ind_list = pd.DataFrame(np.c_[neg_ind_list,[0]*len(neg_ind_list)],columns=["ind","target"])
        ind_list = pd.concat([pos_ind_list,neg_ind_list],axis=0).reset_index(drop=True)

        # shuffle ind_list
        ind_df = ind_list.sample(frac=1)
        batch_size = 64
        g = self._ts_generator_CNN(dataset,ind_df)
        model,num_t = self._fit_on_batch(model,g,batch_size)
        self.model = model
        logging.debug("updating on minibatch completed, total %s loops"%num_t)

    def predict_on_tick(self,df,tick_ind):
        # do prediction on raw tick data
        try:
            model = self.model
        except:
            raise Exception("model has to be trained before doing prediction")
        dataset = df.copy()
        td = pd.Timedelta(seconds=3)
        # generate features
        dataset,feature_list = self._generate_feature(dataset)
        dataset = dataset.dropna()
        ind_b = tick_ind - self.backward_window * td
        if dataset.loc[ind_b:tick_ind].shape[0] < self.window+1:
            y_pred = 0.0
        else:
            ts = dataset.loc[ind_b:tick_ind]
            unique_list = [ts[col].unique().shape[0] for col in ts.columns]
            if max(unique_list) <= 1: # find all same tick
                # logging.debug("find all same ticks, set ypred as 0.0")
                y_pred = 0.0
            else:
                ft_list = pd.DataFrame()
                for ft in feature_list:
                    k_ = (ts[ft] - ts[ft].iloc[0])[1:] / np.arange(1,len(ts))
                    point_time = tick_ind.strftime("%Y%m%d_%H:%M:%S")
                    ft_list["%s_k_%s"%(ft,point_time)] = k_

                # if one col is all the same, transform them as all-zero col
                x = ((ft_list - ft_list.min())/(ft_list.max()-ft_list.min())).fillna(0.0).values
                x = x.reshape(1,x.shape[0],x.shape[1],1)
                y_pred = model.predict(x)[0][1] # prediction is 1D of 2 length vector
        return y_pred



