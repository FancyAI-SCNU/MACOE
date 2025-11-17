import numpy as np
import os
import pandas as pd

from queue import Queue
from datetime import datetime
import time
import pickle

np.random.seed(11)

import sys
ZERO = 1e-7
sys.path.append("..")


def random_array(p, size):
    
    if type(size) == int:
        size = (size, )
    array_size = np.prod(size)
    num_ones = int(p * array_size)
    array = np.zeros(array_size)
    array[:num_ones] = 1
    np.random.shuffle(array)
    return array.reshape(size)

def toArray(data):
    if type(data) == np.ndarray:
        return data

    elif type(data) == list:
        data = np.array(data)
        return data

    elif type(data) == pd.DataFrame:
        share_index = toArray(data.index)
        share_value = toArray(data.values)
        share_colmns = toArray(data.columns)
        return share_index, share_value, share_colmns

    else:
        try:
            share_array = np.array(data)
            return share_array
        except:
            raise NotImplementedError



class Sampler:
    
    def __init__(self, config):
        self.raw_dir = "./cn_1min/backtest/"
        self.order_dir = "./cn_1min/order/order/train/"
        self.ins_list = [f[:-9] for f in os.listdir(self.order_dir) if f.endswith("target")]
        
         
        self.file_list = ["ind_stock/Agr_Fore_Hus_Fish.txt","ind_stock/arch.txt","ind_stock/car.txt",\
                          "ind_stock/chemical_industry.txt","ind_stock/computer.txt","ind_stock/eat.txt",\
                            "ind_stock/Household_appliances.txt","ind_stock/Light_man.txt",\
                                "ind_stock/med.txt","ind_stock/nonferrous_metal.txt"]
        
        # self.features = config["features"]
        self.features=[{'name':'raw','type':'range','loc':'./cn_1min/normed_feature/','size': 180}]
        self.queue = Queue(10000000)
        self.child = None
        self.ins = None
        self.raw_df = None
        self.df_list = None
        self.order_df = None
        
        


    @staticmethod
    def _worker(order_dir, raw_dir, features, file_list, queue):
        ins = None
        index = 60
        date_list = []
        num_sample = [10,13,15,10,12,14,17,18,10,10]
        
        
        print("sample data start!")
        # while True:
        mark = 0
        ins_list = []
        ins_file_list = file_list
        ff = 0
        for file_name in ins_file_list:
            txt_tables = []
            f = open(file_name, "r",encoding='utf-8')
            line = f.readline() 
            while line:
                txt_tables.append(line.split("\n")[0]) 
                line = f.readline() 
            
            ins_list.append(np.random.choice(txt_tables, num_sample[ff], replace=False))
            ff+=1
        
        ins_list = ins_list[:-2]
        
        ins = ins_list
        feature_df_lists_agents = []
        order_df_agents = []
        raw_dfs_agents = []
        date_lists_agents = []
        
        for k in range(len(ins_list)):
            feature_df_lists = []
            order_df = []
            raw_dfs = []
            date_lists = []
            
            for i in range(len(ins_list[k])):
                
                order = pd.read_pickle(order_dir + ins[k][i] + ".p.target")
                
                order_df.append(order)
                feature_df_list = []
            
                for feature in features:
                    feature_df_list.append(pd.read_pickle(f"{feature['loc']}/{ins[k][i]}.pkl"))
                 
                feature_df_lists.append(feature_df_list)

                raw_df = pd.read_pickle(raw_dir + ins[k][i] + ".pkl.backtest")
                
                 
                date_list = order_df[i].index.get_level_values(0).tolist()
                date_list=date_list[1:]
                raw_dfs.append(raw_df)
                date_lists.append(date_list)
                index = 60
                
                
            feature_df_lists_agents.append(feature_df_lists)
            order_df_agents.append(order_df)
            raw_dfs_agents.append(raw_dfs)
            date_lists_agents.append(date_lists)
            
        date, day_raw_df_value, day_raw_df_column, day_raw_df_index, day_feature_dfs_, \
            target, is_buy = [],[],[],[],[],[],[]
        
        ix_ = 0
        
        while True:
            
            index = np.random.randint(0,len(date_lists_agents[0][0])-1)
            ins_queue = []
            date_queue = []
            day_raw_df_value_queue = []
            day_raw_df_column_queue = []
            day_raw_df_index_queue = []
            day_feature_dfs_queue = []
            
            target_queue = []
            is_buy_queue = []
            amount_temp = []
            result_actions_queue = []

            pre_short_list_agents=[]
            pre_long_list_agents=[]
            
            date_list_, day_raw_df_value, day_raw_df_column, day_raw_df_index, day_feature_dfs_, \
            target, is_buy = [],[],[],[],[],[],[]
            result_actions=[]
            pre_short_list = []
            pre_long_list = []
            for ins_ in range(len(ins_list)):
                if mark==1:
                    break
                
                for j in range(len(ins_list[ins_])):
                    
                    date_ = date_lists_agents[0][0][index]
                    
                    if date_ not in date_lists_agents[ins_][j]:
                        mark=1
                        break
                    
                    
                    date_list_.append(date_)
                    day_order_df = order_df_agents[ins_][j].loc[pd.IndexSlice[date_]]
                    
                    target_ = day_order_df["amount"]
                    target_ = 1000000
                    target.append(target_)
                    amount_temp.append(target_)
                    
                    if target_ < ZERO:
                        print(target_)
                        mark =1
                        break
                    day_feature_dfs = []
                    
                    
                    date_str = str(date_).split(" ")[0]+" 09:30:00"
                    
                    if not (date_ in raw_dfs_agents[ins_][j].index.levels[2]):
                        mark=1
                        break

                    day_raw_df = raw_dfs_agents[ins_][j].loc[pd.IndexSlice[:, :, date_]]
                    
                    ix = 0
                     
                    for df in feature_df_lists_agents[ins_][j]:
                        if ix == 0:
                            
                            df_temp_test = df.index.get_level_values(1).map(lambda x:str(x))
                            order_valid = df[df_temp_test == date_str]
                            
                            day_feature_dfs.append(order_valid.values)
                             
                        else:
                            
                            df_temp_test = df.index.get_level_values(1).map(lambda x:str(x))
                            order_valid = df[df_temp_test == date_list_[i]]
                            day_feature_dfs.append(order_valid.values)
                        
                        ix +=1

                    

                    if np.isnan(order_valid.values).any():
                        df_temp_test = feature_df_lists_agents[ins_][j][0].index.get_level_values(1).map(lambda x:str(x))
                        x= feature_df_lists_agents[ins_][j][0][df_temp_test == date_str]
                        
                        mark=1
                        break

                                
                    day_feature_dfs = np.array(day_feature_dfs)
                   
                    day_raw_df_index_, day_raw_df_value_, day_raw_df_column_ = toArray(day_raw_df)
                    
                    day_raw_df_index.append(day_raw_df_index_)
                    day_raw_df_value.append(day_raw_df_value_)
                    day_raw_df_column.append(day_raw_df_column_)
                    
                    if np.isnan(day_feature_dfs).any():
                        mark = 1
                        break
                    day_feature_dfs_i = toArray(day_feature_dfs)
                    
                    day_feature_dfs_.append(day_feature_dfs_i)
                if mark==0:
                    
                    order_list = np.random.choice([-1, 1], size=400) 
                    for i in range(order_list.shape[0]):
                        is_buy.append(0)
                    
                if mark ==0:
                    
                    ins_queue.append(ins_list[ins_])
                    date_queue.append(date_list_)
                    day_raw_df_value_queue.append(day_raw_df_value)
                    day_raw_df_column_queue.append(day_raw_df_column)
                    day_raw_df_index_queue.append(day_raw_df_index)
                    day_feature_dfs_queue.append(day_feature_dfs_)
                    
                    pre_long_list_agents.append(pre_long_list)
                    pre_short_list_agents.append(pre_short_list)
                     
                    
                    target_queue.append(target)
                    is_buy_queue.append(is_buy)
                    is_buy = []
                    day_raw_df_value = []
                    day_raw_df_column = []
                    day_raw_df_index = []
                    day_feature_dfs_ = []
                    result_actions=[]
                    
                    date_list_ = []
                    pre_long_list = []
                    pre_short_list = []
                    target = []
                    
                   
                else:
                    
                    
                    break
            # index+=1
             
            if mark ==0:
                
                     
                queue.put(
                        (ins_queue, date_queue, day_raw_df_value_queue, day_raw_df_column_queue, \
                        day_raw_df_index_queue, day_feature_dfs_queue, target_queue, is_buy_queue,),
                        block=True,
                    )
                print("sample data finish!")
                break
            else:
                    mark=0
                    
                    continue

    def _sample_ins(self):
        """ """
        return np.random.choice(self.ins_list, 1)[0]

    def reset(self):
        """ """
        self._worker(self.order_dir, self.raw_dir, self.features, self.file_list, self.queue)
        # print("sample data finish!")

    def sample(self):
        if self.queue.empty():
            return None
        return self.queue.get(block=True)


class TestSampler(Sampler):
    
    def __init__(self, config):
        super().__init__(config)
        self.ins_index = -1
        self.order_dir = "./cn_1min/order/order/test/"
        
        self.file_list = ["ind_stock/Agr_Fore_Hus_Fish.txt","ind_stock/arch.txt","ind_stock/car.txt",\
                          "ind_stock/chemical_industry.txt","ind_stock/computer.txt","ind_stock/eat.txt",\
                            "ind_stock/Household_appliances.txt","ind_stock/Light_man.txt",\
                                "ind_stock/med.txt","ind_stock/nonferrous_metal.txt"]
        
    def _sample_ins(self):
        """ """
        self.ins_index += 1
        if self.ins_index >= len(self.ins_list):
            return None
        else:
            return self.ins_list[self.ins_index]

    @staticmethod
    def _worker(order_dir, raw_dir, features, file_list, queue):
        
        print("test data sample start")
        count = 27
        for i in range(1,count+1):
            mark = 0
            with open('./save_test_order1_all_a/data'+str(i)+'.pkl', 'rb') as f:
                loaded_data = pickle.load(f)
                (
                ins_list_temp,
                date,
                day_raw_df_value,
                day_raw_df_column,
                day_raw_df_index,
                day_feature_dfs_,
                target,
                is_buy,
                        ) = loaded_data
           
            if mark == 1:
                continue
            
            if len(ins_list_temp)>8:
                ins_list_new = ins_list_temp[:-2][:]
            else:
                ins_list_new = ins_list_temp
            if len(date)>8:
                date_new = date[:-2][:]
            else:
                date_new = date
            if len(day_raw_df_value)>8:
                day_raw_df_value_new = day_raw_df_value[:-2][:]
            else:
                day_raw_df_value_new = day_raw_df_value
            if len(day_raw_df_column)>8:
                day_raw_df_column_new = day_raw_df_column[:-2][:]
            else:
                day_raw_df_column_new = day_raw_df_column
            if len(day_raw_df_column)>8:
                day_raw_df_index_new = day_raw_df_index[:-2][:]
            else:
                day_raw_df_index_new = day_raw_df_index
            if len(day_feature_dfs_)>8:
                day_feature_dfs_new = day_feature_dfs_[:-2][:]
            else:
                day_feature_dfs_new = day_feature_dfs_
            if len(target)>8:
                target_new = target[:-2][:]
            else:
                target_new = target
            if len(is_buy)>8:
                is_buy_new = is_buy[:-2][:]
            else:
                is_buy_new = is_buy
            
            queue.put(
                        (
                            ins_list_new,
                            date_new,
                            day_raw_df_value_new,
                            day_raw_df_column_new,
                            day_raw_df_index_new,
                            day_feature_dfs_new,
                            target_new,
                            is_buy_new,
                        ),
                        block=True,
                    )
        
        print("test sample finsh!")
        return queue

    def reset(self, order_dir=None):
        
        return self._worker(self.order_dir, self.raw_dir, self.features, self.file_list, self.queue)

