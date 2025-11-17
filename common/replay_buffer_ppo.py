import threading
import numpy as np

import torch
import random

random.seed(11)
np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)

class Buffer:
    def __init__(self):
        self.size = int(5e5)
        self.batch_size = 16
        self.current_size = 0
        
        self.buffer = dict()
        self.n_agents = 8 
       
        self.obs_shape =  156 
        self.action_shape = 5
        self.msg_dims = 80
        self.reward_dim=1
        self.time_step = 8
        
        self.buffer['stock_n_batch'] = []
        self.buffer['o_batch'] = []
        self.buffer['u_batch'] = []
        self.buffer['o_next_batch'] = []
        self.buffer['reward_batch'] = []
        self.buffer['done_batch'] = []
        self.buffer['f_0_batch'] =[]
        self.buffer['f_0_next_batch'] =[]
        self.buffer['pi_batch'] = []
        self.buffer['msg_batch'] =[]
        self.buffer['log_batch'] = []
        self.buffer['log_ref_batch'] = []
        self.buffer['rew_all_agents_batch'] = []
        self.buffer['msg_f_0_batch'] =[]
        self.buffer['msg_pi_batch'] =[]

        self.buffer['stock_n'] = []
        self.buffer['o'] =[]
        self.buffer['u'] = []
        self.buffer['o_next'] =[]
        self.buffer['reward'] = []
        self.buffer['done'] = []
        self.buffer['f_0'] =[]
        self.buffer['f_0_next'] =[]
        self.buffer['pi'] =[]
        self.buffer['msg'] =[]
        self.buffer['log'] =[]
        self.buffer['log_ref'] =[]
        self.buffer['rew_all_agents'] =[]
        self.buffer['msg_f_0'] =[]
        self.buffer['msg_pi'] =[]

         # thread lock
        self.lock = threading.Lock()
        
    # store the episode
    def store_episode_batch(self, o, u, r, o_next,done,f_0,f_0_next,pi,msg_f_0,msg_pi,msg,log,log_ref,time_step):
        # idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        with self.lock:
            stock_n_batch=[]
            o_batch=[]
            u_batch=[]
            reward_batch=[]
            o_next_batch =[]
            done_batch =[]
            log_batch=[]
            log_ref_batch=[]
            msg_batch = []
            f_0_batch=[]
            pi_batch=[]
            
            for n_ in range(self.n_agents):
                
                stock_n_batch.append(o[n_].shape[0])
                o_batch.append(o[n_])# .cpu().numpy()
                u_batch.append(u[n_])
                reward_batch.append(r[n_])
                o_next_batch.append(o_next[n_])
                done_batch.append(done[n_])
                log_batch.append(log[n_])
                log_ref_batch.append(log_ref[n_])
                 
            self.buffer['stock_n_batch'].append(stock_n_batch)
            self.buffer['o_batch'].append(o_batch)# .cpu().numpy()
            self.buffer['u_batch'].append(u_batch)
            self.buffer['reward_batch'].append(reward_batch)
            self.buffer['o_next_batch'].append(o_next_batch)
            self.buffer['done_batch'].append(done_batch)
            self.buffer['log_batch'].append(log_batch)
            self.buffer['log_ref_batch'].append(log_ref_batch)
            # msg f pi 基本不变stock 10->1
            self.buffer['msg_batch'].append(msg)
            self.buffer['f_0_batch'].append(f_0)
            self.buffer['f_0_next_batch'].append(f_0_next)
            self.buffer['pi_batch'].append(pi)
            self.buffer['msg_f_0_batch'].append(msg_f_0)
            self.buffer['msg_pi_batch'].append(msg_pi)
         


    def store_episode(self,rew_all_agents,time_record):

        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        
        with self.lock:
            if time_record<self.time_step:
                for t in range(time_record,self.time_step):
                    self.buffer['o_batch'].append(self.buffer['o_batch'][time_record-1])
                    self.buffer['u_batch'].append(self.buffer['u_batch'][time_record-1])
                    self.buffer['reward_batch'].append(self.buffer['reward_batch'][time_record-1])
                    self.buffer['o_next_batch'].append(self.buffer['o_next_batch'][time_record-1])
                    self.buffer['done_batch'].append(self.buffer['done_batch'][time_record-1])
                    self.buffer['f_0_batch'].append(self.buffer['f_0_batch'][time_record-1])
                    self.buffer['f_0_next_batch'].append(self.buffer['f_0_next_batch'][time_record-1])
                    self.buffer['pi_batch'].append(self.buffer['pi_batch'][time_record-1])
                    self.buffer['msg_batch'].append(self.buffer['msg_batch'][time_record-1])
                    self.buffer['log_batch'].append(self.buffer['log_batch'][time_record-1])
                    self.buffer['log_ref_batch'].append(self.buffer['log_ref_batch'][time_record-1])
                    self.buffer['stock_n_batch'].append(self.buffer['stock_n_batch'][time_record-1])
                    self.buffer['msg_f_0_batch'].append(self.buffer['msg_f_0_batch'][time_record-1])
                    self.buffer['msg_pi_batch'].append(self.buffer['msg_pi_batch'][time_record-1])
                    rew_all_agents.append(rew_all_agents[time_record-1])
            self.buffer['o'].append(self.buffer['o_batch'])
            self.buffer['u'].append(self.buffer['u_batch'])
            self.buffer['reward'].append(self.buffer['reward_batch'])
            self.buffer['o_next'].append(self.buffer['o_next_batch'])
            self.buffer['done'].append(self.buffer['done_batch'])
            self.buffer['f_0'].append(self.buffer['f_0_batch'])
            self.buffer['f_0_next'].append(self.buffer['f_0_next_batch'])
            self.buffer['pi'].append(self.buffer['pi_batch'])
            self.buffer['msg'].append(self.buffer['msg_batch'])
            self.buffer['log'].append(self.buffer['log_batch'])
            self.buffer['log_ref'].append(self.buffer['log_ref_batch'])
            self.buffer['stock_n'].append(self.buffer['stock_n_batch'])
            self.buffer['msg_f_0'].append(self.buffer['msg_f_0_batch'])
            self.buffer['msg_pi'].append(self.buffer['msg_pi_batch'])
            self.buffer['rew_all_agents'].append(rew_all_agents)
        
        self.buffer['o_batch']=[]
        self.buffer['u_batch'] = []
        self.buffer['o_next_batch'] = []
        self.buffer['reward_batch'] = []
        self.buffer['done_batch'] = []
        self.buffer['f_0_batch'] = []
        self.buffer['f_0_next_batch'] = []
        self.buffer['pi_batch'] = []
        self.buffer['msg_batch'] = []
        self.buffer['log_batch'] = []
        self.buffer['log_ref_batch'] =[]
        self.buffer['rew_all_agents_batch'] = []
        self.buffer['stock_n_batch']=[]
        self.buffer['msg_f_0_batch']=[]
        self.buffer['msg_pi_batch']=[]


    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer
    
    def get(self, idx):
        return self.buffer['o'][idx],self.buffer['u'][idx],self.buffer['reward'][idx],\
            self.buffer['o_next'][idx],self.buffer['done'][idx]
    
    def get_size(self):
        return self.current_size
    
    def clear(self):
        if self.current_size>self.size:
            self.buffer['stock_n'] = []
            self.buffer['o'] =[]
            self.buffer['u'] = []
            self.buffer['o_next'] =[]
            self.buffer['reward'] = []
            self.buffer['done'] = []
            self.buffer['f_0'] =[]
            self.buffer['pi'] =[]
            self.buffer['msg'] =[]
            self.buffer['log'] =[]
            self.buffer['log_ref'] =[]
            self.buffer['rew_all_agents'] =[]
    
    def get_all_step(self):
        return self.buffer['o_batch'],self.buffer['u_batch'],self.buffer['reward_batch'],\
            self.buffer['o_next_batch'],self.buffer['done_batch'],self.buffer['f_0_batch'],\
            self.buffer['pi_batch'],self.buffer['msg_batch'],self.buffer['log_batch'],self.buffer['log_ref_batch']
    
    def get_all(self):
        transitions = []
        
        temp_buffer = {}
        print(self.current_size,"zzzz")
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][:self.current_size]
            
        return temp_buffer
    def get_batch_ppo(self):
        
        print(self.current_size)
        start = max(0, self.current_size - self.batch_size)
        idxs = list(range(start, self.current_size))
        batch = {}
        for key, vals in self.buffer.items():
            if '_batch' not in key:
               
                batch[key] = [vals[i % len(vals)] for i in idxs]
        return batch
    
    def get_batch(self):
        
        available = min(self.current_size, self.size)
        if available < self.batch_size:
            raise ValueError(f"Not enough samples: {available} < {self.batch_size}")
        
        idxs = random.sample(range(available), self.batch_size)
        batch = {}
        for key, vals in self.buffer.items():
            if '_batch' not in key:
                
                batch[key] = [vals[i] for i in idxs]
        return batch
    
    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
