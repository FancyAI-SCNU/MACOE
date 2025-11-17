import pickle
import pandas as pd
import numpy as np
import os

path = "./cn_1min/backtest/"
path_list = os.listdir(path)
save_path = "./cn_1min/order/all/"
for i in range(len(path_list)):
    path_ = path+path_list[i]
    with open(path_,'rb') as f:
        data = pickle.load(f)
     
    start = 0
    end = 1
    df = data.groupby('datetime').take(range(start, end)).droplevel(level=0)


    div = df['$volume0'].rolling((end - start)*60).mean().shift(1).groupby(level='datetime').transform('first')

    order = df.groupby(level=(2)).mean().dropna()
    order = pd.DataFrame(order)

    order['amount'] = np.random.lognormal(-3.28, 1.14) * order['$volume0']
    order['order_type'] = 0
    order = order.drop(columns=["$volume0", "$vwap0"])
    
    save_name = path_list[i].replace(".backtest",".target")
     
    with open(save_path+save_name,'wb') as f:
        order.to_pickle(f)
    print(order.shape,save_path)
