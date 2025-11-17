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
        label_lists_agents = []
        
        for k in range(len(ins_list)):
            feature_df_lists = []
            order_df = []
            raw_dfs = []
            date_lists = []
            label_lists = []
            for i in range(len(ins_list[k])):
                
                order = pd.read_pickle(order_dir + ins[k][i] + ".p.target")
                
                order_df.append(order)
                feature_df_list = []
            
                for feature in features:
                    feature_df_list.append(pd.read_pickle(f"{feature['loc']}/{ins[k][i]}.pkl"))
                
                feature_df_lists.append(feature_df_list)

                raw_df = pd.read_pickle(raw_dir + ins[k][i] + ".pkl.backtest")
                
                date_list = order_df[i].index.get_level_values(0).tolist()
                
                raw_dfs.append(raw_df)
                date_lists.append(date_list)
                index = 60
            feature_df_lists_agents.append(feature_df_lists)
            order_df_agents.append(order_df)
            raw_dfs_agents.append(raw_dfs)
            date_lists_agents.append(date_lists)
            
        
        date, day_raw_df_value, day_raw_df_column, day_raw_df_index, day_feature_dfs_, \
            target, is_buy = [],[],[],[],[],[],[]
        label_dfs_ = []
        ix_ = 0
        
        while True:
            
            index = np.random.randint(0,len(date_lists_agents[0][0])-1)
            ins_queue = []
            date_queue = []
            day_raw_df_value_queue = []
            day_raw_df_column_queue = []
            day_raw_df_index_queue = []
            day_feature_dfs_queue = []
            label_dfs_queue = []
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
                    label_dfs_ = []
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
        
        ins_list = []
        for file_name in file_list:
            txt_tables = []
            f = open(file_name, "r",encoding='utf-8')
            line = f.readline() 
            while line:
                txt_tables.append(line.split("\n")[0]) 
                line = f.readline() 
            ins_list.append(txt_tables)
        
        order_df = pd.read_pickle(order_dir + ins_list[0][0] + ".p.target")
        date_list = order_df.index.get_level_values(0).tolist()
        print(len(date_list))
        print("test data sample start")
        
        
        count = 0
        for index in range(len(date_list)):
           
            mark = 0
           
            while True:
                ins_list_temp_agents = []
                date_agents = []
                target_agents = []
                is_buy_agents = []
                amount_temp_agents = []
                pre_long_list_agents = []
                pre_short_list_agents = []
                ins_id_ = 0
                day_raw_df_value_agents,day_raw_df_column_agents,day_raw_df_index_agents,\
                    day_feature_dfs_agents = [],[],[],[]
                
                ix__ = 0
                if mark ==1:
                    ins_list = []
                    for file_name in file_list:
                        txt_tables = []
                        f = open(file_name, "r",encoding='utf-8')
                        line = f.readline() 
                        while line:
                            txt_tables.append(line.split("\n")[0]) 
                           
                            line = f.readline() 
                        ins_list.append(txt_tables)
                    mark = 0
                
                    ins_list = ins_list[:-2]
                
                for agent in range(len(ins_list)):
                    ins_list_temp = []
                    date_lists = []
                    target = []
                    is_buy = []
                    amount_temp = []
                    pre_long_list = []
                    pre_short_list = []
                    day_raw_df_value,day_raw_df_column,day_raw_df_index,day_feature_dfs_ = [],[],[],[]
                    for stock in range(len(ins_list[agent])):
                        
                        ins_list_temp.append(ins_list[agent][stock])
                        order_df = pd.read_pickle(order_dir + ins_list[agent][stock] + ".p.target")
                        df_list = []
                        for feature in features:
                            df_list.append(pd.read_pickle(f"{feature['loc']}/{ins_list[agent][stock]}.pkl"))
                        
                        raw_df = pd.read_pickle(raw_dir + ins_list[agent][stock] + ".pkl.backtest")
                        date_list_now = order_df.index.get_level_values(0).tolist()
                        
                        date_ = date_list[index]
                        date_str = str(date_).split(" ")[0]+" 09:30:00"
                    
                        
                        if date_ not in date_list_now:
                            mark = 1
                            print("11111")
                            break
                        date_lists.append(date_)
                        day_df_list = []
                        
                       
                        day_raw_df = raw_df.loc[pd.IndexSlice[:, :, date_]]
                        
                        day_order_df = order_df.loc[pd.IndexSlice[date_]]

                        target_ = day_order_df["amount"]
                        if target_ < 0:
                            mark=1
                            print("22222")
                            print(ins_list[agent][stock])
                            continue
                        target.append(target_)
                        amount_temp.append(target_)
                       
                        ix = 0
                       
                        for df in df_list:
                            if ix == 0:
                                df_temp_test = df.index.get_level_values(1).map(lambda x:str(x))
                                order_valid = df[df_temp_test == date_str]
                                day_df_list.append(order_valid.values)
                                
                                
                            else:
                                df_temp_test = df.index.get_level_values(0).map(lambda x:str(x))
                                order_valid = df[df_temp_test == date_str]
                                day_df_list.append(order_valid.values)
                            ix +=1
                        if np.isnan(order_valid.values).any():
                            print(ins_list[agent][stock])
                            print("33333")
                            mark=1
                           
                            continue
                        
                        day_feature_dfs = np.array(day_df_list)
                        
                        day_raw_df_index_, day_raw_df_value_, day_raw_df_column_ = toArray(day_raw_df)
                        
                        day_feature_dfs_t = toArray(day_feature_dfs)
                        
                        day_raw_df_index.append(day_raw_df_index_)
                        day_raw_df_value.append(day_raw_df_value_)
                        day_raw_df_column.append(day_raw_df_column_)
                        day_feature_dfs_.append(day_feature_dfs_t)
                        
                    
                    if mark==0:
                        order_list = np.random.choice([-1, 1], size=len(ins_list[agent])) 
                        
                        
                        for i in range(order_list.shape[0]):
                            
                            is_buy.append(0)
                    

                    if mark == 0:
                        ix__+=1
                        print(ix__)
                        
                        ins_list_temp_agents.append(ins_list_temp)
                        date_agents.append(date_list)
                        day_raw_df_value_agents.append(day_raw_df_value)
                        day_raw_df_column_agents.append(day_raw_df_column)
                        day_raw_df_index_agents.append(day_raw_df_index)
                        day_feature_dfs_agents.append(day_feature_dfs_)
                        
                        target_agents.append(target)
                        is_buy_agents.append(is_buy)
                    if mark == 1:continue
                    
                
                if mark==0:
                    count+=1
                    print(len(ins_list_temp_agents),"2222")
                    data=( ins_list_temp_agents,
                        date_agents,
                        day_raw_df_value_agents,
                        day_raw_df_column_agents,
                        day_raw_df_index_agents,
                        day_feature_dfs_agents,
                        target_agents,
                        is_buy_agents)
                    print(len(day_feature_dfs_agents[0]),len(day_raw_df_column_agents[1]))
                    
                    with open('./save_test_order1_all_a1/data'+str(count)+'.pkl','wb') as f:
                        pickle.dump(data, f)
                    print("finish!",count)
                    break
                else:
                    continue
            
        print("test sample finsh!")
        return queue

    def reset(self, order_dir=None):
         
        return self._worker(self.order_dir, self.raw_dir, self.features, self.file_list, self.queue)



if __name__ == "__main__":
    x = TestSampler(0)
    cc = x.reset()
    