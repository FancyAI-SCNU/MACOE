import numpy as np
import torch
import os
# ind change
from maddpg.train_fun_am_mla_gru import MADDPG

np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)

class Agent(torch.nn.Module):
    def __init__(self, 
                agent_id, 
                n_agents, 
                x, 
                a, 
                args, 
                m=0,
                hidden_sizes=(300,300),
                hidden_sizes_model=(128,64,128), 
                action_scale=1, 
                activation="tanh",
                output_activation="tanh", 
                msg_dim=4, is_training=np.bool_(0), 
                max_est_time=5, msg_idx=0,device=None):
        super(Agent, self).__init__()
        self.args = args
        self.agent_id = agent_id
        
        self.policy = MADDPG(agent_id=agent_id,n_agents=n_agents, x=x, a=a,args=args,m=m,
                 hidden_sizes=hidden_sizes,hidden_sizes_model=hidden_sizes_model,
                 action_scale = action_scale,
                 activation=activation,output_activation=output_activation,
                 msg_dim=msg_dim,is_training=is_training,
                 max_est_time=max_est_time, msg_idx=msg_idx,device=device
                        )
        

    
    def learn(self, transitions,ix,msg):
        return self.policy.train(transitions,ix,msg)
    
    def learn_single(self, transitions,ix,msg):
        return self.policy.train_single(transitions,ix,msg)
    
    def learn_ppo(self, transitions,ix,msg):
        return self.policy.train_ppo(transitions,ix,msg)

    def learn_ppo_new(self, transitions,ix,msg=None):
        return self.policy.train_ppo_new(transitions,ix,msg)
    
    def msg_generation(self, transitions,ix):
        return self.policy.msg_gen(transitions,ix)

    
    
    def select_action(self, x_agent,m,o,u,agent_ix,f_0_list,pi_list,noise_rate, epsilon=None):
        
        pi,feature,pi_ref = self.policy.first_get_action(x_agent,m,o,u,f_0_list,pi_list,agent_ix)
        u = pi.cpu().detach().numpy()
        
        self.high_action = 1 
        noise = noise_rate * self.high_action * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.high_action, self.high_action)

        u_ref = pi_ref.cpu().detach().numpy()
        u_ref += noise
        u_ref = np.clip(u_ref, -self.high_action, self.high_action)
        
        return u,feature,u_ref,pi
    
    def save_model(self,train_step,ix,model_path):
        self.policy.save_model(train_step,ix,model_path)
    
    def save_best_model_single(self,train_step,ix,model_path):
        self.policy.save_best_model_single(train_step,ix,model_path)

    def save_best_model(self,train_step,ix,model_path):
        self.policy.save_best_model(train_step,ix,model_path)
    
    def load_best_model(self,ix,model_path):
        self.policy.load_best_model(ix,model_path)

     
    def soft_update(self):
        self.policy._soft_update_target_network()
 
    def print_grad(self):
        self.policy.print_grad()
