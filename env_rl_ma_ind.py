

import gym

gym.logger.set_level(40)
import numpy as np
import pandas as pd
import time

import copy

import reward
import observation
import action
import copy
from common.utils_ma_ind import Sampler

import torch.nn.functional as F
import torch
ZERO = 1e-7

np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)

def nan_weighted_avg(vals, weights, axis=None):
    """

    :param vals: The values to be averaged on.
    :param weights: The weights of weighted avrage.
    :param axis: On which axis to calculate the weighted avrage. (Default value = None)

    """
    assert vals.shape == weights.shape, AssertionError(f"{vals.shape} & {weights.shape}")
    vals = vals.copy()
    weights = weights.copy()
    res = (vals * weights).sum(axis=axis) / weights.sum(axis=axis)
    return np.nan_to_num(res, nan=vals[0])

def deep_update(
    original, new_dict, new_keys_allowed=False, whitelist=None, override_all_if_type_changes=None,
):
    """Updates original dict with values from new_dict recursively.
    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the whitelist, then new subkeys can be introduced.

    :param original: Dictionary with default values.
    :type original: dict
    :param new_dict(dict: dict): Dictionary with values to be updated
    :param new_keys_allowed: Whether new keys are allowed. (Default value = False)
    :type new_keys_allowed: bool
    :param whitelist: List of keys that correspond to dict
    values where new subkeys can be introduced. This is only at the top
    level. (Default value = None)
    :type whitelist: Optional[List[str]]
    :param override_all_if_type_changes: List of top level
    keys with value=dict, for which we always simply override the
    entire value (dict), iff the "type" key in that value dict changes. (Default value = None)
    :type override_all_if_type_changes: Optional[List[str]]
    :param new_dict:

    """
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}` ".format(k))

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if (
                k in override_all_if_type_changes
                and "type" in value
                and "type" in original[k]
                and value["type"] != original[k]["type"]
            ):
                original[k] = value
            # Whitelisted key -> ok to add new subkeys.
            elif k in whitelist:
                deep_update(original[k], value, True)
            # Non-whitelisted key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original

def merge_dicts(d1, d2):
    """

    :param d1: Dict 1.
    :type d1: dict
    :param d2: Dict 2.
    :returns: A new dict that is d1 and d2 deep merged.
    :rtype: dict

    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged

class StockEnv(gym.Env):
    """Single-asset environment"""
    def __init__(self, config):
        
        self.max_step_num = 236
        self.limit = 1
        self.time_interval = 30
        self.interval_num = 8

        self.offset =  0
        self.last_reward = None
    
        self.log = "./example/"
        obs_conf = {}
        features=[{'name':'raw','type':'range','loc':'./cn_1min/normed_feature','size': 180}]
        obs_conf["features"] = features
        obs_conf["time_interval"] = self.time_interval
        obs_conf["max_step_num"] = self.max_step_num
        self.obs = getattr(observation, "TeacherObs")(obs_conf)
        act_conf = {}
        # change
        act_conf["action_num"] = 5
        
        act_conf["action_map"] = [0, 0.125, 0.25, 0.375, 0.5]
        self.action_map_ = act_conf["action_map"]
       
        self.action_func = getattr(action, "Static_Action")(act_conf)
        
        self.action_func = getattr(action, "Static_Action")(act_conf)
        self.reward_func_list = []
        self.reward_log_dict = {}
        self.reward_coef = []
        rew_conf = {"VP_Penalty_small_vec":{"penalty":5,"coefficient":1}}
        for name, conf in rew_conf.items():
            
            self.reward_coef.append(conf.pop("coefficient"))
            self.reward_func_list.append(getattr(reward, name)(conf))
            self.reward_log_dict[name] = 0.0
        
        self.number = []
        self.n_agents = 8
        self.sum_this_ffr=0.0
        self.sum_performance_raise = 0.0
        self.sum_PA = 0.0
        self.sum_this_vwap = 0.0
         
        
         

    def toggle_log(self, log):
        self.log = log

    # reset sigle agent
    def reset(self, sample):
        """
        :param sample:

        """
        self.sum_this_ffr=0.0
        self.sum_performance_raise = 0.0
        self.sum_PA = 0.0
        self.sum_this_vwap = 0.0
        self.arr = 0.0
        # self.mark = np.zeros([self.n_agents,self.number])
        for key in self.reward_log_dict.keys():
            self.reward_log_dict[key] = 0.0
        
        
        if not sample is None:
            (
                self.ins_list,
                self.date_list,
                self.raw_df_values_list,
                self.raw_df_columns_list,
                self.raw_df_index_list,
                self.feature_dfs_list,
                self.target_list,
                self.is_buy_list
            ) = sample
        
        
        self.state_list_agents = []
        self.this_valid_list_agents = []
        self.raw_df_list_agents = []
        self.this_cash_list_agents = []
        self.day_vwap_list_agents = []
        self.day_twap_list_agents = []
        self.done_list_agents = []

        self.result_agents =[]
        self.mark = []
        self.number = []
        # self.position_list = self.target_list.copy()
        self.position_list = copy.deepcopy(self.target_list)
        
        for ins_ in range(len(self.ins_list)):
            self.state_list = []
            self.this_valid_list = []
            self.raw_df_list = []
            self.this_cash_list = []
            self.day_vwap_list = []
            self.day_twap_list = []
            self.done_list = []
            self.market_data_list = []
            self.number.append(len(self.ins_list[ins_]))
            self.mark.append(np.zeros((len(self.ins_list[ins_]))))
            for i in range(len(self.ins_list[ins_])):
                self.raw_df = pd.DataFrame(index=self.raw_df_index_list[ins_][i], \
                                           data=self.raw_df_values_list[ins_][i], \
                                            columns=self.raw_df_columns_list[ins_][i],)
                
                start_time = time.time()
                self.load_time = time.time() - start_time
                self.day_vwap = nan_weighted_avg(
                    self.raw_df["$vwap0"].values[self.offset : self.offset + self.max_step_num],
                    self.raw_df["$volume0"].values[self.offset : self.offset + self.max_step_num],
                )
                
                try:
                    assert not (np.isnan(self.day_vwap) or np.isinf(self.day_vwap))
                except:
                    print(self.raw_df)
                    print(self.ins_list[ins_][i])
                    print(self.day_vwap)
                    self.raw_df.to_pickle("/nfs_data1/kanren/error_df.pkl")
                self.day_vwap_list.append(self.day_vwap)
                self.day_twap = np.nanmean(self.raw_df["$vwap0"].values[self.offset : self.offset + self.max_step_num])
                
                self.day_twap_list.append(self.day_twap)
                self.t = -1 + self.offset
                self.interval = 0
                self.position = self.target_list[ins_][i]
                self.eps_start = time.time()
                 
                self.state = self.obs(
                    self.raw_df,
                    self.feature_dfs_list[ins_][i],
                    self.t,
                    self.interval,
                    self.position,
                    self.target_list[ins_][i],
                    self.is_buy_list[ins_][i],
                    self.max_step_num,
                    self.interval_num,
                    i,
                )
                 
                market_data = self.state[:6*236]
                market_data = market_data.reshape(-1,6)
                add = np.zeros((4,6))
                market_data = np.concatenate((market_data,add))
                self.market_data = market_data.reshape(8,30,6).mean(1)
                self.market_data_list.append(self.market_data)

                self.state = np.concatenate([self.state[6*236+6*226:-19],self.state[-19:-1]])
                
                position_emb = np.concatenate((np.zeros((60,)),np.ones((18,))))
                self.state = np.concatenate([self.state,position_emb])
                
                
                self.state_list.append(self.state)
           
                self.done = False
                self.done_list.append(self.done)
                if self.limit > 1:
                    self.this_valid = np.inf
                    self.this_valid_list.append(self.this_valid)
                else:
                    self.this_valid = np.nansum(self.raw_df["$volume0"].values) * self.limit
                    self.this_valid_list.append(self.this_valid)
                self.this_cash = 0
                self.this_cash_list.append(self.this_cash)
                self.raw_df_list.append(self.raw_df)


            self.state_list_agents.append(torch.tensor(self.state_list))
            self.this_valid_list_agents.append(self.this_valid_list)
            self.raw_df_list_agents.append(self.raw_df_list)
            self.this_cash_list_agents.append(self.this_cash_list)
            self.day_vwap_list_agents.append(self.day_vwap_list)
            self.day_twap_list_agents.append(self.day_twap_list)
            self.done_list_agents.append(torch.tensor(self.done_list))

        
         
        self.step_time = []
        self.action_log = [np.nan] * self.interval_num
        self.reset_time = time.time() - start_time
        self.real_eps_time = self.reset_time
        self.total_reward = 0
        self.total_instant_rew = 0
        self.last_rew = 0
        #
        self.count_act = []
        self.penalty_act = []
        self.logit_store = []
        self.all_reward_step = []
        for n_i in range(self.n_agents):
           
            self.count_act.append(np.zeros([self.state_list_agents[n_i].shape[0]]))
            self.penalty_act.append(np.zeros([self.state_list_agents[n_i].shape[0],8]))
            self.logit_store.append(np.zeros([self.state_list_agents[n_i].shape[0],8]))
            self.all_reward_step.append(np.zeros([self.state_list_agents[n_i].shape[0],8]))
         
        
        self.max_count = 4
        self.ix = 0
        self.pa_t = [0.0]*8
        
        return self.state_list_agents
    


    def step(self, action,action_logit,done_mod=None):
        start_time = time.time()
       
        self.action_log[self.interval] = action
        
        reward_list = []
        plenty_smooth_list = []
        
        info_list = []
        order = 0
         
        all_agents_up_list = []
        all_agents_down_list = []
        
        ix=0
        if done_mod is not None:
            self.done_list_agents = done_mod
        
        for agent in range(len(action)):
            all_agents_up_list.append(np.zeros((self.number[agent])))
            all_agents_down_list.append(np.zeros((self.number[agent])))
            reward_single = []
            
            for single_ in range(action[agent].shape[0]):
                ix+=1
                
                if done_mod is None:
                    volume_t = self.action_func(
                        action[agent][single_],
                        self.target_list[agent][single_],
                        self.position_list[agent][single_],
                        max_step_num=self.max_step_num,
                        t=self.t - self.offset,
                        interval=self.interval,
                        interval_num=self.interval_num,
                    )
                    
                    
                    if action[agent][single_]!=0:
                        order += self.time_interval
                  
                    if self.done_list_agents[agent][single_]:
                        self.logit_store[agent][single_,self.interval] = 0.0
                    else:
                        
                        logit_1 = 0#action_logit[agent][single_][0]
                        logit_2 = 0#np.sum(action_logit[agent][single_][1:])
                        
                        logit_ = torch.tensor([logit_1,logit_2])
                        if torch.sum(logit_)!=0:
                            action_logit_ = logit_/torch.sum(logit_)
                        else:
                            action_logit_ = logit_
                            
                        action_0 = action_logit_[0]
                        action_1 = 1-action_0
                         
                        self.logit_store[agent][single_,self.interval] = action_1
                   
                    reward = 0.0
                    time_left = self.max_step_num - self.t - 1 + self.offset
                    
                    time_left = min(self.time_interval, time_left)
                    
                    v_t = np.repeat(volume_t / time_left, time_left)
                    
                    # 30
                    minutes = np.arange(self.t + 1, self.t + time_left + 1)
                    
                    
                    vwap_t = self.raw_df_list_agents[agent][single_].iloc[minutes]["$vwap0"].values
                    vol_t = self.raw_df_list_agents[agent][single_].iloc[minutes]["$volume0"].values
                    max_vol_t = self.limit * vol_t if self.limit < 1 else np.inf
                    
                    v_t = np.minimum(v_t, max_vol_t)
                    
                    
                    if v_t[0] > ZERO:
                        
                        self.count_act[agent][single_]+=1
                        
                        pen = F.relu(torch.tensor(self.count_act[agent][single_]-self.max_count))/8.
                        self.penalty_act[agent][single_,self.interval]+= self.action_map_[action[agent][single_]]
                    if self.t + time_left == self.max_step_num - 1 + self.offset:
                        left = self.position_list[agent][single_] - v_t.sum()
                        
                        v_t[-1] += left
                        v_t = np.minimum(v_t, max_vol_t)
                        
                        
                    this_money = (v_t * vwap_t).sum()
                    this_vol = v_t.sum()
                    this_vwap = np.nan_to_num(this_money / this_vol)
                    
                    
                    self.t += time_left
                    self.interval += 1
                    
                    
                    self.position_list[agent][single_] -= this_vol
                    
                    self.this_cash_list_agents[agent][single_] += this_money
                   
                    all_agents_up_list[agent][single_]=this_vwap
                    all_agents_down_list[agent][single_]=self.day_twap_list_agents[agent][single_]
                    if self.is_buy_list[agent][single_]:
                        performance_raise = (1 - this_vwap / self.day_vwap_list_agents[agent][single_]) * 10000
                        
                        PA_t = (1 - this_vwap / self.day_twap_list_agents[agent][single_]) * 10000
                    else:
                        performance_raise = (this_vwap / self.day_vwap_list_agents[agent][single_] - 1) * 10000
                        PA_t = (this_vwap / self.day_twap_list_agents[agent][single_] - 1) * 10000
                    self.pa_t[self.interval-1]+=PA_t
                    
                    
                    for i, reward_func in enumerate(self.reward_func_list):
                        if reward_func.isinstant:
                            
                           
                            tmp_r = reward_func(performance_raise, v_t, self.target_list[agent][single_], PA_t)
                            reward += tmp_r * self.reward_coef[i]
                            
                            
                            self.all_reward_step[agent][single_,self.interval-1] = reward
                            self.reward_log_dict[type(reward_func).__name__] += tmp_r
                            
                    if this_vol.all() == 0.0:
                        reward -= 0.4*(performance_raise * v_t.sum() / self.target_list[agent][single_])
                        

                    if self.position_list[agent][single_] <= ZERO:
                        
                        self.done_list_agents[agent][single_] = True

                    if self.interval == self.interval_num:
                         
                        self.done_list_agents[agent][single_] = True
                     
                    self.step_time.append(time.time() - start_time)
                    self.real_eps_time += time.time() - start_time
                
                if done_mod is None:
                    reward_single.append(reward)
                if self.done_list_agents[agent][single_]:
                    
                    this_traded = self.target_list[agent][single_] - self.position_list[agent][single_]
                   
                    this_vwap = (self.this_cash_list_agents[agent][single_] / this_traded) \
                        if this_traded > ZERO else self.day_vwap_list_agents[agent][single_]
                    
                    valid = min(self.target_list[agent][single_], self.this_valid_list_agents[agent][single_])
                     
                    this_ffr = (this_traded / valid) if valid > ZERO else 1.0
                    if abs(this_ffr - 1.0) < ZERO:
                        this_ffr = 1.0
                    this_ffr *= 100
                    
                    this_vv_ratio = this_vwap / self.day_vwap_list_agents[agent][single_]
                   
                    vwap = self.raw_df_list_agents[agent][single_]["$vwap0"].values[self.offset : self.max_step_num + self.offset]
                    this_tt_ratio = this_vwap / np.nanmean(vwap)
                    
                    if self.is_buy_list[agent][single_]:
                        performance_raise = (1 - this_vv_ratio) * 10000
                        PA = (1 - this_tt_ratio) * 10000
                    else:
                        performance_raise = (this_vv_ratio - 1) * 10000
                        PA = (this_tt_ratio - 1) * 10000
                    
                    if done_mod is None:
                        for i, reward_func in enumerate(self.reward_func_list):
                            if reward_func.isinstant:
                                
                                
                                tmp_r = 0 #40 * (((self.target_list[agent][single_]-this_traded) / self.target_list[agent][single_]) ** 2).sum()
                               
                                if agent>=len(reward_list):
                                    reward_single[single_]-=tmp_r
                                else:
                                    reward_list[agent][single_] -= tmp_r
                            
                    if self.log:
                        
                        res = pd.DataFrame(
                            {
                                "target": self.target_list[agent][single_],
                                "sell": not self.is_buy_list[agent][single_],
                                "vwap": this_vwap,
                                "this_vv_ratio": this_vv_ratio,
                                "this_ffr": this_ffr,
                            },
                            
                            index=[self.ins_list[agent][single_], self.date_list[agent][0]],
                        )
                    money = self.target_list[agent][single_] * self.day_vwap
                
                    if self.is_buy_list[agent][single_]:
                        info = {
                            "money": money,
                            "money_buy": money,
                            "action": self.action_log,
                            "ffr": this_ffr,
                            "obs0_PR": performance_raise,
                            "ffr_buy": this_ffr,
                            "PR_buy": performance_raise,
                            "PA": PA,
                            "PA_buy": PA,
                            "vwap": this_vwap,
                        }
                    else:
                        info = {
                            "money": money,
                            "money_sell": money,
                            "action": self.action_log,
                            "ffr": this_ffr,
                            "obs0_PR": performance_raise,
                            "ffr_sell": this_ffr,
                            "PR_sell": performance_raise,
                            "PA": PA,
                            "PA_sell": PA,
                            "vwap": this_vwap,
                        }
                    
                    if self.mark[agent][single_]==0:
                        self.ix+=1
                        
                        self.sum_PA+=PA
                        
                        self.sum_performance_raise+=performance_raise
                        self.sum_this_ffr+=this_ffr
                        self.sum_this_vwap+=this_vwap
                        self.mark[agent][single_]=1
                        
                    
                    if self.log:
                        
                        info["res"] = res
                    info_list.append(info)
                    

                else:
                    
                    state_temp = torch.tensor(self.obs(
                        self.raw_df_list_agents[agent][single_],
                        self.feature_dfs_list[agent][single_],
                        self.t,
                        self.interval,
                        self.position_list[agent][single_],
                        self.target_list[agent][single_],
                        self.is_buy_list[agent][single_],
                        self.max_step_num,
                        self.interval_num,
                        agent,
                        action[agent][single_],
                        
                    ))
                  
                    
                    if self.t<10:
                    
                        state_temp = torch.cat((state_temp[6*236+6*226+self.t*6:-19],state_temp[:self.t*6],\
                                                                state_temp[-19:-1]))
                        
                        position_emb = torch.cat((torch.zeros((10*6-self.t*6,)),torch.ones((self.t*6+18,)),))
                    else:
                        
                        state_temp = torch.cat((state_temp[self.t*6-6*10:self.t*6],\
                                                                state_temp[-19:-1]))
                        
                        position_emb = torch.ones(10*6+18,)
                    
                    
                    self.state_list_agents[agent][single_] = torch.cat((state_temp,position_emb))
                    
                
                 
                if ix!=sum(self.number):
                
                    self.interval-=1
                   
                    self.t -= time_left
                result_list = (self.sum_PA, self.sum_performance_raise ,self.sum_this_ffr \
                               ,self.sum_this_vwap,self.ix)
            reward_list.append(reward_single)    
            
       
        reward_all = 0.0
        
        
        self.pa_t[self.interval-1]=self.pa_t[self.interval-1]/sum(self.number)
        return self.state_list_agents, reward_list, self.done_list_agents, result_list,\
            self.penalty_act,self.logit_store,self.all_reward_step,\
                self.count_act,reward_all,self.pa_t

 