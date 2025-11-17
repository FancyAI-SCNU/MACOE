import pickle
import numpy
import pandas as pd
import os

path = "demo/qlib_data_cn_normalize_demo/"
path_list = os.listdir(path)
save_path = "./cn_1min/backtest/"
max_step_num=1
for i in range(len(path_list)):
    path_ = path + path_list[i]
    feature = pd.read_csv(path_)
    feature['VWAP'] = (feature['OPEN'] + 2*feature['HIGH'] + 2*feature['LOW'] + feature['CLOSE'])/6

    feature = feature.fillna(0)
    def select(name):
        pau_null = ~feature['OPEN'].isnull()
        
        select_fea = feature[name][pau_null]

        return select_fea

    def fillnan(col):
        col=col.fillna(method="ffill")
        col=col.fillna(method="bfill")
        return col

    def IF():
        right = select('VWAP')
        is_null = right.isnull()
        left = fillnan(select('CLOSE'))
        result = pd.Series(numpy.where(is_null, left, right),index=is_null.index)
       
        return result

    def Cut_1():
        
        feature_cut = fillnan(select('CLOSE'))
        feature_cut = feature_cut.iloc[max_step_num : None]
        
        return feature_cut

    def Cut_2():
        
        feature_cut = IF()
        feature_cut = feature_cut.iloc[max_step_num : None]
        
        return feature_cut

    def if_2():
        gt = numpy.greater(select('VWAP'),numpy.multiply(1.001,select('HIGH')))
        lt = numpy.less(select('VWAP'),numpy.multiply(0.999,select('LOW')))
        or_ = numpy.bitwise_or(gt,lt)
        if_ = pd.Series(numpy.where(or_, 0, select('VOLUME')),index=or_.index)
        return if_

    def if_1():
        if_ = pd.Series(numpy.where(select('VOLUME').isnull(), 0, if_2()),index=select('VOLUME').isnull().index)
        return if_

    def Cut_3():
        feature_cut = if_1()
        feature_cut = feature_cut.iloc[max_step_num : None]
        return feature_cut

    df =pd.DataFrame()
    df["$close0"] = Cut_1()
    df["$vwap0"] = Cut_2()
    df["$volume0"] = Cut_3()
    df['date'] = feature['date']

    df["datetime"]=df["date"]
    for j in range(0,len(df['date']),max_step_num):
        
        df['datetime'].iloc[j:j+max_step_num] = pd.to_datetime(str(df['date'].iloc[j]).split(" ")[0])
    
    df.set_index(["date","datetime"], append=True, drop=True, inplace=True)
    
    save_list = save_path+path_list[i].replace(".csv",".pkl.backtest")
     
    with open(save_list,'wb') as f:
        df.to_pickle(f)
    