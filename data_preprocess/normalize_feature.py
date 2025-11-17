import numpy
import pandas as pd
import pickle
import os
 
 
path = "demo/qlib_data_cn_normalize_demo/"
path_list = os.listdir(path)
save_path = "./cn_1min/normed_feature/"

for i in range(len(path_list)):
    path_ = path + path_list[i]
    feature = pd.read_csv(path_)
    feature['VWAP'] = (feature['OPEN'] + 2*feature['HIGH'] + 2*feature['LOW'] + feature['CLOSE'])/6

    def select(name):
        pau_null = feature['PAUSED'].isnull()
        pau_eq = numpy.equal(feature['PAUSED'],0.0)
        pau_or = numpy.bitwise_or(pau_null, pau_eq)
        select_fea = feature[name][pau_or]

        return select_fea

    def fillnan(col):
        col=col.fillna(method="ffill")
        col=col.fillna(method="bfill")
        return col

    def IF(name):
        right = select(name)
        is_null = right.isnull()
        left = fillnan(select('CLOSE'))
        result = pd.Series(numpy.where(is_null, left, right),index=is_null.index)
       
        return result

    def DayLast(col):
        for j in range(0,feature.shape[0],236):
            col[j:j+236] = col[j+236-1]
        return col

    def Ref(col,N):
        
        col = col.shift(N)
        
        return col

    def Cut(name,shift=None):
        if shift is None:
            
            feature_cut = IF(name)/Ref(DayLast(feature['CLOSE']),236)
            
            feature_cut = feature_cut.iloc[236 : None]
            
            
        else:
           
            feature_cut = Ref(IF(name),shift)/Ref(DayLast(feature['CLOSE']),236)
            feature_cut = feature_cut.iloc[236 : None]
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

    def volume(shift=None):
        
        mean_ = getattr(if_1().rolling(7080, min_periods=1), "mean")()
        
        ref_ = Ref(DayLast(mean_),236)
        
        if shift is None:
            cut_ = if_1()/ref_
            vol = cut_.iloc[236 : None]
        else:
            cut_ = Ref(if_1(),shift)/ref_
            vol = cut_.iloc[236 : None]
        
        return vol
    

    df = pd.DataFrame()
    df['$open'] = Cut('OPEN')
    df['$high'] = Cut('HIGH')
    df['$low'] = Cut('LOW')
    df['$close'] = Cut('CLOSE')
    df['$vwap'] = Cut('VWAP')
    
    df['$open_1'] = Cut('OPEN',236)
    df['$high_1'] = Cut('HIGH',236)
    df['$low_1'] = Cut('LOW',236)
    df['$close_1'] = Cut('CLOSE',236)
    df['$vwap_1'] = Cut('VWAP',236)
    
    df["$volume"] = volume()
    df["$volume_1"] = volume(236)
    
    EPS = 1e-12
    df_values = df.values
    names = {
        "price": slice(0, 10),
        "volume": slice(10, 12),
    }
    feature_med = {}
    feature_std = {}
    feature_vmax = {}
    feature_vmin = {}
    for name, name_val in names.items():
        part_values = df_values[:, name_val].astype(numpy.float32)
        if name == "volume":
            part_values = numpy.log1p(part_values)
        feature_med[name] = numpy.nanmedian(part_values)
        part_values = part_values - feature_med[name]
        feature_std[name] = numpy.nanmedian(numpy.absolute(part_values)) * 1.4826 + EPS
        part_values = part_values / feature_std[name]
        feature_vmax[name] = numpy.nanmax(part_values)
        feature_vmin[name] = numpy.nanmin(part_values)
    
    df["date"] = pd.to_datetime(
        feature["date"][236:].values
    )
    
    dx = df["date"].values
    dx = dx[::236]
    dx = pd.DataFrame(dx)
   
    dx = dx.reset_index(drop=True)

    df.set_index("date", append=True, drop=True, inplace=True)
    df_values = df.values

    names = {
        "price": slice(0, 10),
        "volume": slice(10, 12),
    }
    
     
    for name, name_val in names.items():
        if name == "volume":
            df_values[:, name_val] = numpy.log1p(df_values[:, name_val])
        df_values[:, name_val] -= feature_med[name]
        df_values[:, name_val] /= feature_std[name]
        slice0 = df_values[:, name_val] > 3.0
        slice1 = df_values[:, name_val] > 3.5
        slice2 = df_values[:, name_val] < -3.0
        slice3 = df_values[:, name_val] < -3.5

        df_values[:, name_val][slice0] = (
            3.0 + (df_values[:, name_val][slice0] - 3.0) / (feature_vmax[name] - 3) * 0.5
        )
        df_values[:, name_val][slice1] = 3.5
        df_values[:, name_val][slice2] = (
            -3.0 - (df_values[:, name_val][slice2] + 3.0) / (feature_vmin[name] + 3) * 0.5
        )
        df_values[:, name_val][slice3] = -3.5
    idx = df.index
    
    idx = idx[::236]
    
    idx.set_names(["Instrument","datetime"], inplace=True)
    
    
    feat = df_values[:, [0, 1, 2, 3, 4, 10]].reshape(-1, 6 * 236)
    feat_1 = df_values[:, [5, 6, 7, 8, 9, 11]].reshape(-1, 6 * 236)
    
    df_new_features = pd.DataFrame(
        data=numpy.concatenate((feat, feat_1), axis=1),
        
        columns=["FEATURE_%d" % k for k in range(12 * 236)],
    )
    
    
    df_new_features["date"] = dx
    df_new_features.set_index("date", append=True, drop=True, inplace=True)
    
    save_list = save_path+path_list[i].replace(".csv",".pkl")
     
    with open(save_list,'wb') as f:
        df_new_features.to_pickle(f)
