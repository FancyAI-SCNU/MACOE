import numpy as np
import pandas as pd
import os
import time
import datetime
from joblib import Parallel, delayed

data_path = './cn_1min/order/'

in_dir = os.path.join(data_path, 'all/')

def generate_order(df, start, end):

    df = df.groupby('date').take(range(start, end)).droplevel(level=0)
    div = df['$volume0'].rolling((end - start)*60).mean().shift(1).groupby(level='date').transform('first')
    order = df.groupby(level=(2, 0)).mean().dropna()
    
    order = pd.DataFrame(order)
    
    order['amount'] = np.random.lognormal(-3.28, 1.14) * order['$volume0']
    order['order_type'] = 0
    order = order.drop(columns=["$volume0", "$vwap0"])
    return order

def w_order(f, start, end):
    print("start,111")
    order = pd.read_pickle(in_dir + f)
    
    df_temp = order.index.get_level_values(0).map(lambda x:str(x))
    
    order_train = order.iloc[:3,:]
    
    order_test = order.iloc[3:,:]
     
    # order_train = order[df_temp <= '2022-01-06']
    
    # order_test = order[df_temp >= '2022-01-07']
    
    # df_temp = order_train.index.get_level_values(0).map(lambda x:str(x))
    # order_train = order_train[df_temp >= '2022-01-01']

    # df_temp_test = order_test.index.get_level_values(0).map(lambda x:str(x))

    # df_temp_test = order_test.index.get_level_values(0).map(lambda x:str(x))
     
    # order_test = order_test[df_temp_test <= '2022-01-10']
    
    if len(order_train) > 0:
        print("train")
        print(train_path + f[:-9] + '.target')
        order_train.to_pickle(train_path + f[:-9] + '.target')
    if len(order_test) > 0:
        print("test")
        order_test.to_pickle(test_path + f[:-9] + '.target')
     
    print("finish,111")
    return 0

train_path = os.path.join(data_path, "order_1/train/")
if not os.path.exists(train_path):
    os.makedirs(train_path)


test_path = os.path.join(data_path, "order_1/test/")
if not os.path.exists(test_path):
    os.makedirs(test_path)

 
res = Parallel(n_jobs=6)(delayed(w_order)(f, 0, 235) for f in os.listdir(in_dir))
print(sum(res))
