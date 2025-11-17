from tqdm import tqdm
from agent_am_mla_gru import Agent

import torch
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy

from torch.optim.lr_scheduler import MultiStepLR

from actor_critic_gru import MLA
from common.utils_ma_ind import Sampler,TestSampler
from env_rl_ma_ind import StockEnv
np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)
import warnings
warnings.filterwarnings("ignore")

from trl.grpo_base import GRPOTrainer
from trl import GRPOConfig

from actor_critic_am_mla import Q_fun

import math

from common.replay_buffer_ppo import Buffer as ppo_buffer

import json

torch.autograd.set_detect_anomaly(True)

block_size = 128

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        
        super(RMSNorm, self).__init__()
        
        self.eps = eps

    def forward(self, x,gamma):
        
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  
        
        x_norm = x / rms  
        
        return gamma * x_norm  

class Runner:
    def __init__(self, args,env):

        self.args = args
        # CUDA
        
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        self.gpus = [4,5]
        self.max_time_step = 8
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.noise = self.args.noise_rate
        self.epsilon = self.args.epsilon
        self.n_agents = 8 
        # change
        self.env = env
        self.msg_dims = self.args.msg_dim
        self.obs_dim = 78*2 
        self.act_dim = 5
        
        self.agent_stocks = 10 


        self.gamma= self.args.gamma

        self.agents = self._init_agents()
         
        self.batch_size = self.args.batch_size
        self.epoch = self.args.epoch

        self.return_ = -1000000000.0
        self.PA_ = -1000000000.0
        self.count = 0
        self.GLR = 0
        
        self.softmax_scale = 0.6
        self.ppo_buffer = ppo_buffer()
        self.q__ = True
       
        self.norm = RMSNorm(self.msg_dims)
        
        mla_layers = 1
        self.mla  = nn.ModuleList(
            [MLA(self.device).to(self.device) for _ in range(mla_layers)]
        )
         
        MLA_params = []
        self.gamma_ = nn.Parameter(torch.ones(self.msg_dims)).to(self.device)
        for pname,p in self.mla.named_parameters():
            MLA_params += [p]
        self.mla_param = MLA_params
       
        milestones = [30, 60, 120, 200]
       
        lr = 8e-4
        self.mla_optimizer = torch.optim.Adam(MLA_params, lr=lr)
        self.mla_scheduler = MultiStepLR(
            self.mla_optimizer, milestones=milestones, gamma=0.8)
        lr_q = 8e-4
       
        self.q_critic = Q_fun(n_agents=self.n_agents,x=32,a=self.act_dim,activation="relu",hidden_sizes=(32,32)).to(self.device)
        self.q_critic_target = Q_fun(n_agents=self.n_agents,x=32,a=self.act_dim,activation="relu",hidden_sizes=(32,32)).to(self.device)
        if self.q__:
            Q_params = []
            for qname, q in self.q_critic.named_parameters():
                # print(qname,q.numel())
                if 'mlp_cen' in qname:
                    Q_params += [q]
                
            self.Q_optimizer = torch.optim.Adam(Q_params, lr=lr_q)
            self.Q_scheduler = MultiStepLR(
                self.Q_optimizer, milestones=milestones, gamma=0.8)
            self.q_critic_target.load_state_dict(self.q_critic.state_dict())
            self.tau = self.args.tau
        
        self.q_param = Q_params
        self.grpo = True
        self.test = False
        self.ppo = False
        self.conntinue = True
        self.xuanzhuan_loss = False

        self.path = "./model_harmo/"
        self.pretrain_path = "./model_pretrain/"


    def _init_agents(self):
        
        x_shape = self.obs_dim   
        a_shape = self.act_dim  
        m = self.msg_dims  
        hidden_sizes = (256,256)
        hidden_sizes_model = (128,64,128)
        action_scale=1
        activation="tanh"
        output_activation="tanh"
        msg_dim = self.msg_dims
        is_training = np.bool_(0)
        max_est_time = 5
        msg_idx = 5
        agents = []
        for i in range(self.n_agents):
            agent = Agent(agent_id=i,n_agents=self.n_agents, 
                x=x_shape, 
                a=a_shape, 
                args=self.args, 
                m=m, 
                hidden_sizes=hidden_sizes,
                hidden_sizes_model=hidden_sizes_model, 
                action_scale=action_scale, 
                activation=activation,
                output_activation=output_activation, 
                msg_dim=msg_dim, is_training=is_training, 
                max_est_time=max_est_time, msg_idx=msg_idx,device=self.device).to(self.device)
            agents.append(agent)
        return agents
     
    def run(self):
         
        returns = []
        self.evaluate_ep = 6
        self.save_rate = 20
        self.k_round = 1
        sample = Sampler(0)
        test_data = TestSampler(0)
        
        if self.test:
            print("test")
            cc_q_test = test_data.reset()
            for test_a in range(self.n_agents):
                self.agents[test_a].load_best_model(test_a,self.path)
            model_path = self.path
            self.mla.load_state_dict(torch.load(model_path + '/' + '7mla_params.pkl'))
            self.q_critic.load_state_dict(torch.load(model_path + '/' + '7q_params.pkl'))
            
            return_,PA,order_ = self.evaluate(cc_q_test)
            exit()
        if self.conntinue:
            print("loading model...")
            for con_a in range(self.n_agents):
                self.agents[con_a].load_best_model(con_a,self.pretrain_path)
             
            
        self.ma_update_num = 2
        self.ma_update = 0
        current_size = 0
        for ep in range(self.epoch):
            sample.reset()
            sample_queue = sample.sample()
            state = self.env.reset(sample_queue)
            msg_prev = np.zeros([self.msg_dims])
            msg_prev = torch.tensor(msg_prev).float().to(self.device)
            msg_pprev = np.zeros([self.msg_dims])
            msg_pprev = torch.tensor(msg_pprev).float().to(self.device)

            u = []
            for i_u in range(self.n_agents):
                u.append(torch.tensor(np.zeros([state[i_u].shape[0],self.act_dim])).to(self.device))
            
            f_0_input_all_agents  = []
            
            pi_list_all_agents = []
            
            f_0_input_list = []
            pi_list = []
            for n in range(self.n_agents):
                
                f_0_input = torch.zeros([1,32])
                
                f_0_input = torch.tensor(f_0_input).float().to(self.device)
                pi_ = torch.zeros([5])
                pi_ = torch.tensor(pi_).float().to(self.device)
                 
                f_0_input_list.append(f_0_input.unsqueeze(0))
                pi_list.append(pi_.unsqueeze(0))
                
                
            f_0_input_all_agents = torch.concat(f_0_input_list)
            pi_list_all_agents = torch.concat(pi_list)
            
            
            step = 0
            reward_smooth = 0.0
            reward_all = 0.0
            reset = 1
            
            pi_losses=[]
           
            done_train = False
            ppo_t_record = 0
            while True:
                
                if step>1 and done_train:
                    break
                input =state 
                
                msgs_t = []
                actions_t_logits = []
                actions_t_logits_ref = []
                actions_t = []

                f_0_input_all_agents_new = []
                pi_all_agents_new = []
                with torch.no_grad():
                     
                    mm=0
                    for mla in self.mla:
                        if mm==0:
                            msg = mla(f_0_input_all_agents,pi_list_all_agents,msg_prev)
                        else:
                            msg = mla(f_0_input_all_agents,pi_list_all_agents,self.norm(msg,self.gamma_))
                        mm+=1
                     
                    for i in range(self.n_agents):
                       
                        action,feature,action_ref,pi_= self.agents[i].select_action(input[i].float().to(self.device), msg,input,\
                                                                       u[i].float(),i,f_0_input_all_agents,pi_list_all_agents,\
                                                                        0.1)
                         
                        action = action.squeeze(0)
                        action_ref = action_ref.squeeze(0)
                        actions_t_logits.append(action)
                        actions_t_logits_ref.append(action_ref)

                        act = torch.argmax(torch.tensor(action), dim=-1)
                        actions_t.append(act)
                        
                        
                        f_0_input_new = torch.mean(feature,dim=1) 
                     
                        pi_new = torch.tensor(pi_)
                        pi_new = torch.mean(pi_new,dim=1)
                        
                        f_0_input_all_agents_new.append(f_0_input_new.unsqueeze(0))
                        pi_all_agents_new.append(pi_new)

                   
                    f_0_input_all_agents_new = torch.concat(f_0_input_all_agents_new,dim=0)
                    pi_all_agents_new = torch.concat(pi_all_agents_new,dim=0)
                    
                
                f_0_input_all_agents_new = f_0_input_all_agents_new
                pi_all_agents_new = pi_all_agents_new

                msg_f_0 = f_0_input_all_agents
                msg_pi = pi_list_all_agents
                f_0_input_all_agents = f_0_input_all_agents_new
                pi_list_all_agents = pi_all_agents_new
                
                input_next,reward,done_list,_info,penalty_act,logit,rew_all,count,rew_agents,rew_mark\
                      = self.env.step(actions_t,actions_t_logits)
                
                with torch.no_grad():
                    mm=0
                    for mla in self.mla:
                        if mm==0:
                            msg_ = mla(f_0_input_all_agents,pi_list_all_agents,msg)
                        else:
                            msg_ = mla(f_0_input_all_agents,pi_list_all_agents,self.norm(msg_,self.gamma_))
                        mm+=1
                    actions_t_logits_next = []
                    actions_t_logits_ref_next = []
                    actions_t_next =[]
                    f_0_input_all_agents_new_next = []
                    pi_all_agents_new_next=[]
                    for i in range(self.n_agents):
                            
                        action_next,feature_next,action_ref_next,pi_next= self.agents[i].select_action(input_next[i].float().to(self.device), msg_,input_next,\
                                                                    u[i].float(),i,f_0_input_all_agents,pi_list_all_agents,\
                                                                        0.1)
                        
                        action_next = action_next.squeeze(0)
                        action_ref_next = action_ref_next.squeeze(0)
                        actions_t_logits_next.append(action_next)
                        actions_t_logits_ref_next.append(action_ref_next)

                        act_next = torch.argmax(torch.tensor(action_next), dim=-1)
                        actions_t_next.append(act_next)
                        
                       
                        f_0_input_new_next = torch.mean(feature_next,dim=1) 
                       
                        pi_new_next = torch.tensor(pi_next)
                        pi_new_next = torch.mean(pi_new_next,dim=1)
                        
                        f_0_input_all_agents_new_next.append(f_0_input_new_next.unsqueeze(0))
                        pi_all_agents_new_next.append(pi_new)

                    f_0_input_all_agents_new_next = torch.concat(f_0_input_all_agents_new_next,dim=0)
                    pi_all_agents_new_next = torch.concat(pi_all_agents_new_next,dim=0)
                    
              
                for i in range(len(input_next)):
                    for j in range(len(input_next[i])):
                        if input_next[i][j].shape[0]!=self.obs_dim:
                            input_next[i][j]=input_next[i][j][:self.obs_dim]
                   
                pred_logprobs=[]
                pred_logprobs_ref=[]
                for log_n in range(self.n_agents):
                    reward[log_n]=torch.tensor(reward[log_n])
                    actions_t_logits[log_n] = torch.tensor(actions_t_logits[log_n])
                    actions_t_logits_ref[log_n] = torch.tensor(actions_t_logits_ref[log_n])
                    pred_logprobs_ = F.log_softmax(actions_t_logits[log_n], dim=-1)
                    pred_logprobs_ref_ = F.log_softmax(actions_t_logits_ref[log_n], dim=-1)
                    pred_logprobs.append(pred_logprobs_)
                    pred_logprobs_ref.append(pred_logprobs_ref_)
                
                done_list_tensor = []
                for done_1 in range(len(done_list)):
                    done_list_tensor.append(done_list[done_1])
               
                if reset == 1:
                    
                    self.ppo_buffer.store_episode_batch(input,actions_t_logits,reward,input_next,done_list_tensor,f_0_input_all_agents,f_0_input_all_agents_new_next\
                                                        ,pi_list_all_agents,msg_f_0,msg_pi,msg_prev,pred_logprobs,pred_logprobs_ref,ppo_t_record)
                    ppo_t_record+=1
                else:
                    
                    self.ppo_buffer.store_episode_batch(input,actions_t_logits,reward,input_next,done_list_tensor,f_0_input_all_agents,f_0_input_all_agents_new_next\
                                                        ,pi_list_all_agents,msg_f_0,msg_pi,msg_prev,pred_logprobs,pred_logprobs_ref,ppo_t_record)
                    ppo_t_record+=1
                 
                msg_prev = msg.clone()
                 
                state = input_next
                reset = 0
                
                
                p=0
                p_tar = 0
                for d_1 in range(len(done_list)):
                    for d_2 in range(len(done_list[d_1])): 
                        p_tar +=1
                        if done_list[d_1][d_2]:
                            p+=1
                
                if p_tar == p:
                    
                    done_train = True
                    
                    time_sums = [ pa.sum(axis=0)     
                                for pa in penalty_act ]

                   
                    time_sums = np.concatenate(time_sums, axis=0) 
                    current_step = time_sums.mean(axis=0)     

                    agent_var = current_step.var()  
                    time_sums = [ pa.mean(axis=1)     
                                for pa in penalty_act ]

                    time_sums = np.concatenate(time_sums, axis=0)
                    
                    stock_var = current_step.var()  
                    max_count = 980.0
                    penalty_count_ = 0
                    for p_n in range(self.n_agents):
                        penalty_count_+= np.sum(count[p_n])
                    
                    min_count = 640.0
                    penalty_count = F.relu(torch.tensor(penalty_count_-max_count))/(p_tar*4)
                    penalty_count_min = F.relu(torch.tensor(min_count-penalty_count_))/(p_tar*4)
                    
                    
                    input,actions_t_logits,reward,input_next,done\
                            ,f_0_list,pi_list_,msg_,log,log_ref= self.ppo_buffer.get_all_step()
                    
                    reward_all_agents = reward[:]
                    
                    for t_r in range(len(reward_all_agents)):
                        for n_r in range(self.n_agents):
                            reward_all_agents[t_r][n_r] -=  (agent_var+stock_var)*0.01
                            
                            reward_all_agents[t_r][n_r] -= penalty_count.numpy()*0.8
                            reward_all_agents[t_r][n_r] -= penalty_count_min.numpy()*0.8
                        
                    
                    self.ppo_buffer.store_episode(reward_all_agents,ppo_t_record)
                     
                    ppo_t_record = 0

                     
                    sample.reset()
                    
                    sample_queue = sample.sample()
                    state = self.env.reset(sample_queue)

                    msg_prev = np.zeros([self.msg_dims])
                    msg_prev = torch.tensor(msg_prev).float().to(self.device)
                    msg_pprev = np.zeros([ self.msg_dims])
                    msg_pprev = torch.tensor(msg_pprev).float().to(self.device)
                   
                    reset = 1
                    
                    f_0_input_all_agents  = []
                    
                    pi_list_all_agents = []
                    
                    f_0_input_list = []
                    pi_list = []
                    for n in range(self.n_agents):
                        
                        
                        f_0_input = torch.zeros([1,32])
                        
                        f_0_input = torch.tensor(f_0_input).float().to(self.device)
                        pi_ = torch.zeros([5])
                        pi_ = torch.tensor(pi_).float().to(self.device)
                        
                        f_0_input_list.append(f_0_input.unsqueeze(0))
                        pi_list.append(pi_.unsqueeze(0))
                        
                    f_0_input_all_agents = torch.concat(f_0_input_list)
                    pi_list_all_agents = torch.concat(pi_list)
                fix_batch_size = self.batch_size
                current_batch_size = self.ppo_buffer.get_size()
               
                 
                if (current_batch_size-current_size) >=self.batch_size//4  \
                      and current_batch_size>= self.batch_size and done_train:
                    self.mla.train()
                    self.q_critic.train()
                    current_size = current_batch_size
                    step += 1
                    
                    
                    pi= []
                    pi_target = []
                    pi_opts = []
                    pi_schs = []
                    
                    grpo_losses = [0.0]*8

                    q_losses = [0.0]*8
                    
                    ix = 0

                    self.mla_optimizer.zero_grad()  
                    print("loss start")
                     
                     
                    train_ppo_batch = self.ppo_buffer.get_batch()
                    
                    if self.ppo:
                        agent_id = 0
                        for agent in self.agents:
                            res = agent.learn_ppo_new(train_ppo_batch,agent_id)
                            agent_id+=1
                    if self.grpo:
                        msg_list = []
                        for time_step in range(self.max_time_step):
                            
                            f_0_train=[]
                            pi_train=[]
                            m=[]
                            for ccc in range(len(train_ppo_batch['msg_f_0'])):

                                f_0_train.append(train_ppo_batch['msg_f_0'][ccc][time_step].unsqueeze(0))
                                pi_train.append(train_ppo_batch['msg_pi'][ccc][time_step].unsqueeze(0))
                                m.append(train_ppo_batch['msg'][ccc][time_step].unsqueeze(0))
                            f_0_train = torch.cat(f_0_train).to(self.device)
                            pi_train = torch.cat(pi_train).to(self.device) 
                             
                            m = torch.cat(m).to(self.device) 
                            
                            mm=0
                            for mla in self.mla:
                                if mm==0:
                                    msg = mla(f_0_train,pi_train,m)
                                else:
                                    msg = mla(f_0_train,pi_train,self.norm(msg,self.gamma_))
                                mm+=1 
                            msg_list.append(msg.unsqueeze(0))
                        msg = torch.concat(msg_list).permute(1,0,2)
                         
                        
                        if len(msg.shape)<3:
                            msg = msg.unsqueeze(1)
                        for agent in self.agents:
                            
                            pi_agent,pi_agent_target,\
                            pi_optimizer,pi_scheduler,\
                            loss_grpo = agent.learn(train_ppo_batch,ix,msg)
                           
                            pi.append(pi_agent)
                            pi_target.append(pi_agent_target)
                            
                            grpo_losses[ix]+=loss_grpo
                            
                            pi_opts.append(pi_optimizer)
                            pi_schs.append(pi_scheduler)

                            ix += 1
                     
                    if self.xuanzhuan_loss:
                         
                        pi_xuanzhuan = []
                        for pi_x in pi: 
                            pc = []
                            for c in range(len(pi_x)):
                                 
                                pc.append(pi_x[c].unsqueeze(0))
                            pi_x = torch.cat(pc) 
                            
                             
                            pi_xuanzhuan.append(pi_x)  

                        eps = 1e-8
                        penalty_radius = 1
                        coverage_strength = 0.05

                        all_agent_trade_prob = []
                        penalty = 0
                        entropy_list = []

                        for pi_soft in pi_xuanzhuan:
                            
                            trade_prob = 1 - pi_soft[..., 0]  
                           
                            trade_prob = trade_prob.permute(1, 0, 2)

                            agent_trade_prob = trade_prob.mean(dim=1) 
                            agent_trade_prob = agent_trade_prob / (agent_trade_prob.sum(dim=1, keepdim=True) + eps)
                            all_agent_trade_prob.append(agent_trade_prob) 

                            single_prob = trade_prob.permute(0, 2, 1)
                            for r in range(1, penalty_radius + 1):
                                left = torch.roll(single_prob, shifts=r, dims=-1)
                                right = torch.roll(single_prob, shifts=-r, dims=-1)
                                penalty =penalty+ torch.sum(single_prob * (left + right))  

                            sum_agent = single_prob.sum(dim=-1, keepdim=True)
                            norm_probs = single_prob / (sum_agent + eps)
                            row_entropy = - (norm_probs * torch.log(norm_probs + eps)).sum(dim=-1) 
                            entropy_list.append(row_entropy.mean())

                        if all_agent_trade_prob:
                            flat_probs = torch.cat(all_agent_trade_prob, dim=-1)  
                            coverage_loss = torch.mean((flat_probs - 1.0) ** 2)
                        else:
                            coverage_loss = torch.tensor(0.0, device=pi[0].device)

                        if entropy_list:
                            loss_row = torch.stack(entropy_list).mean()
                        else:
                            loss_row = torch.tensor(0.0, device=pi[0].device)

                        total_loss = (coverage_strength * coverage_loss ) * 0.1
                         

                    if self.ppo!=True:
                        o_list_batch = []
                        u_list_batch = []
                        r_list_batch = []
                        o_next_list_batch= []
                        r_list_agents_batch = []
                        f_0_batch = []
                        f_0_next_batch = []
                        reward_all = 0.0
                        for batch_num in range(fix_batch_size):
                           
                            for time_step in range(self.max_time_step):
                                o_list, u_list ,r_list,r_list_agents,o_next_list,f_0_list,f_0_next_list = [], [],[],[],[],[],[]
                                for aa in range(self.n_agents):
                                    o_list.append(train_ppo_batch['o'][batch_num][time_step][aa].mean(axis=0,keepdim=True).unsqueeze(0))#.float().to(self.device)
                                    u_list.append(train_ppo_batch['u'][batch_num][time_step][aa].mean(axis=0,keepdim=True).unsqueeze(0)) #.float().to(self.device)
                                    r_list.append(train_ppo_batch['reward'][batch_num][time_step][aa].mean(axis=0,keepdim=True).unsqueeze(0)) #.float().to(self.device)
                                    r_list_agents.append(train_ppo_batch['rew_all_agents'][batch_num][time_step][aa].mean(axis=0,keepdim=True).unsqueeze(0)) #.float().to(self.device)
                                    
                                    f_0_list.append(train_ppo_batch['f_0'][batch_num][time_step][aa].unsqueeze(0))
                                    f_0_next_list.append(train_ppo_batch['f_0_next'][batch_num][time_step][aa].unsqueeze(0))
                                    o_next_list.append(train_ppo_batch['o_next'][batch_num][time_step][aa].mean(axis=0,keepdim=True).unsqueeze(0)) #.float().to(self.device)
                                o_list = torch.cat(o_list).float().to(self.device)
                                u_list = torch.cat(u_list).float().to(self.device)
                                r_list = torch.cat(r_list).float().to(self.device)
                                r_list_agents = torch.cat(r_list_agents).float().to(self.device)
                                o_next_list = torch.cat(o_next_list).float().to(self.device)
                                f_0_list = torch.cat(f_0_list).float().to(self.device)
                                f_0_next_list = torch.cat(f_0_next_list).float().to(self.device)
                                 
                                reward_all += torch.mean(r_list)

                                
                                r_list_agents_batch.append(r_list_agents)
                                o_list_batch.append(o_list)
                                u_list_batch.append(u_list)
                                r_list_batch.append(r_list)
                                o_next_list_batch.append(o_next_list)
                                f_0_batch.append(f_0_list)
                                f_0_next_batch.append(f_0_next_list)
                        reward_all = reward_all/(fix_batch_size*self.max_time_step)
                        
                            
                        agent_id = 0
                        pi_losses=[0.0]*8
                        
                        self.Q_optimizer.zero_grad()
                        for agent in self.agents:
                            for batch_num in range(fix_batch_size*self.max_time_step):
                                with torch.no_grad():
                                    pi_other = pi[:]
                                    pi_target_other = pi_target[:]
                                    for xx in range(len(pi_other)):
                                        
                                        pi_other_clone = pi_other[xx][batch_num%self.max_time_step][batch_num//self.max_time_step].detach().clone()
                                         
                                        if xx==0:
                                            pi_other_ = pi_other_clone.mean(axis=0,keepdim=True).unsqueeze(1)
                                        else:
                                            pi_other_ = torch.concat([pi_other_,pi_other_clone.mean(axis=0,keepdim=True).unsqueeze(1)],dim=1)
                                    for xx in range(len(pi_target_other)):
                                        pi_other_target_clone = pi_target_other[xx][batch_num%self.max_time_step][batch_num//self.max_time_step].detach().float()
                                        if xx==0:
                                            pi_target_other_ = pi_other_target_clone.mean(axis=0,keepdim=True).unsqueeze(1)
                                        else:
                                            pi_target_other_ = torch.concat([pi_target_other_,pi_other_target_clone.mean(axis=0,keepdim=True).unsqueeze(1)],dim=1)
                                    
                                pi_ = pi[agent_id][batch_num%self.max_time_step][batch_num//self.max_time_step].mean(axis=0,keepdim=True).float().to(self.device)
                                pi_target_ = pi_target[agent_id][batch_num%self.max_time_step][batch_num//self.max_time_step].mean(axis=0,keepdim=True).float().to(self.device)
                                
                            
                                q,q_pi = self.q_critic(f_0_batch[batch_num],u_list_batch[batch_num],pi_,pi_other_,agent_id)
                                q_target,q_pi_target = self.q_critic_target(f_0_next_batch[batch_num],u_list_batch[batch_num],pi_target_,\
                                                                        pi_target_other_,agent_id)
                                
                                
                                with torch.no_grad():
                                    
                                    backup= r_list_agents_batch[batch_num][agent_id,:]+ self.gamma*q_pi_target
                                
                                q_loss = torch.mean((q-backup)**2)
                                 
                               
                                q_losses[agent_id]+=q_loss
                              
                                pi_loss = -torch.mean(q_pi)
                                if pi_loss.isnan().any():
                                    print(batch_num%self.max_time_step,batch_num//self.max_time_step)
                                    print(pi_loss)
                                else:
                                    pi_losses[agent_id]+=pi_loss
                                
                            agent_id+=1
                                
                           
                        q_losses = [x / (fix_batch_size*self.max_time_step) for x in q_losses]
                        pi_losses = [x / (fix_batch_size*self.max_time_step) for x in pi_losses]        
                        
                        self.ma_update+=1
                        
                        if self.xuanzhuan_loss:
                            total_loss.backward(retain_graph=True)
                        l1_lambda = 0.001
    
                        l1_norm = sum(p.abs().sum() for p in self.mla.parameters())
                        l1_norm1 = sum(p.abs().sum() for p in self.q_critic.parameters())
                        loss_l1 = l1_lambda*(l1_norm+l1_norm1)
                        loss_l1.backward(retain_graph=True)
                        for agent_id in range(self.n_agents):
                            pi_losses[agent_id].backward(retain_graph=True)
                            grpo_losses[agent_id].backward(retain_graph=True)
                            q_losses[agent_id].backward(retain_graph=True)
                            pi_opts[agent_id].step()
                            
                            pi_schs[agent_id].step()
                            
                        print("loss end")
                        for id,agent in enumerate(self.agents):
                            agent.soft_update()
                            for target_param, param in zip(self.q_critic_target.parameters(), self.q_critic.parameters()):
                                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
                            if ep > 0 and ep % self.save_rate == 0:
                                agent.save_model(ep,id,self.path)
                        if self.ma_update%self.ma_update_num==0:
                            
                            for agent_id in range(self.n_agents):
                                q_losses[agent_id].backward(retain_graph=True)
                            self.mla_optimizer.step()
                             

                            self.Q_optimizer.step()
                         
                       
                    else:
                         
                        print("loss end")
                        for id,agent in enumerate(self.agents):
                            if ep > 0 and ep % self.save_rate == 0:
                                agent.save_model(ep,id,self.path)
                         
                        
                    if self.ppo:
                        print("epoch:",ep)
                        print(res)
                        
                    
                    else:
                        print("epoch:",ep)
                        print("q_loss:",sum(q_losses)/len(q_losses),\
                            "pi_loss:",sum(pi_losses)/len(pi_losses))
                        
                       
             
            if ep>=0 and ep % self.evaluate_ep == 0:
                print("test")
                cc_q_test = test_data.reset()
                return_,PA,order_ = self.evaluate(cc_q_test)
            
                if PA>self.PA_:
                    
                    for i in range(self.n_agents):
                        
                        self.agents[i].save_best_model(ep,i,self.path)
                    model_path = self.path
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(self.mla.state_dict(), model_path + '/'  +str(i)+ 'mla_params.pkl')
                    torch.save(self.q_critic.state_dict(), model_path + '/'  +str(i)+ 'q_params.pkl')
             
                    print("save best",return_)
                    self.return_ = return_
                    self.PA_ = PA
                    self.count = order_
                    
                returns.append(return_)
                 
            
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            np.save('returns.pkl', returns)
        print("best PA_:",self.PA_,"count",self.count)    
      
   
    def evaluate(self,cc_q):
       
        returns = []
        PA_list,performance_raise_list,this_ffr_list,this_vwap_list = [],[],[],[]
        arr_list = []
        data_count = 0.
        act_count = 0.
        
        pa_ = []
        record = 0
        datas = []
        while True:
             
            data_count+=1
            cc = cc_q.get(block=True)
            
            if cc_q.empty():
                break
            
            o, r, ep_ret, ep_len,ep_ret2 = self.env.reset(cc), 0, 0, 0,0
            
            
            msg_prev = np.zeros([self.msg_dims])
          
            rewards = 0
            done = False
            PA_all,performance_raise_all,this_ffr_all,this_vwap_all=0.0,0.0,0.0,0.0
            arr_all=0.0
            u = []
            for i_u in range(self.n_agents):
                u.append(torch.tensor(np.zeros([o[i_u].shape[0],self.act_dim])).to(self.device))

            f_0_input_all_agents  = []
            pi_list_all_agents = []
            
            f_0_input_list = []
            pi_list = []
            for n in range(self.n_agents):
                
                f_0_input = torch.zeros([1,32])
                
                f_0_input = torch.tensor(f_0_input).float().to(self.device)
                pi_ = torch.zeros([5])
                pi_ = torch.tensor(pi_).float().to(self.device)
                f_0_input_list.append(f_0_input.unsqueeze(0))
                pi_list.append(pi_.unsqueeze(0))
                
            
            f_0_input_all_agents = torch.concat(f_0_input_list)#.permute(1,0,2)
            pi_list_all_agents = torch.concat(pi_list)#.permute(1,0,2)

            self.mla.eval()
            self.q_critic.eval()
            time_step=0
            while not done:
                
                actions = []
                actions_t = []
                msgs_t = []
                action_logit = []
                input = o
                
                 
                time_step+=1
                msg_prev = torch.tensor(msg_prev).float().to(self.device)
                 
                with torch.no_grad():
                    
                    mm=0
                    for mla in self.mla:
                        if mm==0:
                            msg = mla(f_0_input_all_agents,pi_list_all_agents,msg_prev)
                        else:
                            
                            msg = mla(f_0_input_all_agents,pi_list_all_agents,self.norm(msg,self.gamma_))
                        mm+=1
                    
                    f_0_input_all_agents_new = []
                    pi_all_agents_new = []
                    for i in range(self.n_agents):
                        
                        action,feature,_,pi\
                            = self.agents[i].select_action(input[i].float().to(self.device), msg,\
                                                                    input,u[i],i,f_0_input_all_agents\
                                                                        ,pi_list_all_agents,0.0)
                        
                        action = action.squeeze(0)
                       
                        act = torch.argmax(torch.tensor(action), dim=-1)
                         
                        actions_t.append(act)
                         
                        f_0_input_new = torch.mean(feature,dim=1) 
                        # 1,11,5
                        pi_new = torch.tensor(pi)
                        pi_new = torch.mean(pi_new,dim=1)
                        
                        f_0_input_all_agents_new.append(f_0_input_new.unsqueeze(0))
                        pi_all_agents_new.append(pi_new)

                        action_logit.append(action)

                    f_0_input_all_agents_new = torch.concat(f_0_input_all_agents_new,dim=0)
                    pi_all_agents_new = torch.concat(pi_all_agents_new,dim=0)
                    
                    
                    
                
                f_0_input_all_agents_new = f_0_input_all_agents_new
                pi_all_agents_new = pi_all_agents_new
                
                o2, r, d , res,penalty_act,logit,rew_all,count_act,rew_agents,rew_mark\
                    =self.env.step(actions_t,action_logit)
                 
                
                act_all = count_act
                
                p=0
                
                p_tar = 0
                for d_1 in range(len(d)):
                    for d_2 in range(len(d[d_1])): 
                        p_tar+=1
                        if d[d_1][d_2]:
                            p+=1
                
                
                if p==p_tar:
                    done =True
                   
                    for rr in range(len(r)):
                        rewards += np.mean(r[rr])
                        act_count += np.sum(act_all[rr])
                    rewards = rewards/len(r)
                    
                    pa_.append(np.array(rew_mark))
                     
                    
                    (PA,performance_raise,this_ffr,this_vwap,iii)=res
                    
                    # if iii!=0:
                    PA_all+=(PA/iii)
                    performance_raise_all+=(performance_raise/iii)
                    this_ffr_all+=(this_ffr/iii)
                    this_vwap_all+=(this_vwap/iii)
                    

                for i in range(len(o2)):
                    for j in range(len(o2[i])):
                        if o2[i][j].shape[0]!=self.obs_dim:
                            o2[i][j]=o2[i][j][:self.obs_dim]
                   
                o = o2
                
                msg_prev = np.copy(msg.cpu().numpy())
                f_0_input_all_agents = f_0_input_all_agents_new
                pi_list_all_agents = pi_all_agents_new
            
            returns.append(rewards)
            PA_list.append(PA_all)
            arr_list.append(arr_all)
            this_ffr_list.append(this_ffr_all)
            this_vwap_list.append(this_vwap_all)
            performance_raise_list.append(performance_raise_all)
        
        
        print('Returns is', np.mean(returns))
        print('PA:', np.mean(PA_list),'per:',np.mean(performance_raise_list),\
              'ffr:',np.mean(this_ffr_list),'vwap:',np.mean(this_vwap_list))
        print("order count:",act_count/data_count)
        print(record,data_count)
        return np.mean(returns),np.mean(PA_list),act_count/data_count
            
            


if __name__ == "__main__":
    
    env = StockEnv(0)
    run =Runner(env)
    run.run()