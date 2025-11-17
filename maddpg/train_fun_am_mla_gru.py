import torch
import os


import numpy as np
from torch.optim.lr_scheduler import MultiStepLR


from trl.grpo_base import GRPOTrainer
from trl import GRPOConfig
from util import to_numpy,to_torch_as

np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)

from actor_critic_am_mla_gru import actor_critic

def _episodic_return(v_s_, rew, done, gamma, gae_lambda):
    """Numba speedup: 4.1s -> 0.057s."""
    returns = np.roll(v_s_, 1)
    done = done.numpy()
    rew = rew.numpy()
    m = (1.0 - done) * gamma
    
    delta = rew + v_s_ * m - returns
    m *= gae_lambda
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae_new = delta[i] + m[i] * gae
        gae = gae_new
        returns[i] += gae
    return returns


class MADDPG(torch.nn.Module):
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
                activation="relu",
                output_activation="tanh", 
                msg_dim=4, is_training=np.bool_(0), 
                max_est_time=5, msg_idx=0,device=None):  
        super(MADDPG, self).__init__()

        self.device = device

        self.max_time_step = 8

        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.n_agents = n_agents
        
        self.tau = self.args.tau
        
        
        self.actor_critic = actor_critic(n_agents=n_agents, x=x, a=a,args=args,m=m,
                 hidden_sizes=hidden_sizes,hidden_sizes_model=hidden_sizes_model,
                 action_scale = action_scale,
                 activation=activation,output_activation=output_activation,
                 msg_dim=msg_dim,is_training=is_training,
                 max_est_time=max_est_time, msg_idx=msg_idx)

        self.actor_critic_target = actor_critic(n_agents=n_agents, x=x, a=a,args=args,m=m,
                 hidden_sizes=hidden_sizes,hidden_sizes_model=hidden_sizes_model,
                 action_scale = action_scale,
                 activation=activation,output_activation=output_activation,
                 msg_dim=msg_dim,is_training=is_training,
                 max_est_time=max_est_time, msg_idx=msg_idx)
        self.actor_critic_ref = actor_critic(n_agents=n_agents, x=x, a=a,args=args,m=m,
                 hidden_sizes=hidden_sizes,hidden_sizes_model=hidden_sizes_model,
                 action_scale = action_scale,
                 activation=activation,output_activation=output_activation,
                 msg_dim=msg_dim,is_training=is_training,
                 max_est_time=max_est_time, msg_idx=msg_idx)
        
        
        lr_1 = 8e-4
        milestones = [30, 60, 120, 200]
        pi_attn_params = []
         
        pi_attn_name = []
         
        
        for qname, q in self.actor_critic.named_parameters():
            # print(qname,q.numel())
            if 'linear_mlp' not in qname:
                pi_attn_params += [q]
                pi_attn_name += [qname]
        
         
        self.pi_attn_params = pi_attn_params
         
        self.pi_attn_name = pi_attn_name
         
        self.pi_optimizer = torch.optim.Adam(pi_attn_params, lr=lr_1)
        self.pi_scheduler = MultiStepLR(
            self.pi_optimizer, milestones=milestones, gamma=0.8)
        
        self.obs_dim = 78*2
        self.act_dim = 5

        self.actor_critic_target.load_state_dict(self.actor_critic.state_dict())
        
        self.grpo_config = self.get_grpo_config()
        self.grpo_trainer = GRPOTrainer(self.grpo_config)

        self.label = False
        self.volume = False

        self._rew_norm = True
        self._lambda = 0.95
        self.dist_fn = torch.distributions.Categorical
        self._eps_clip = 0.2
        self._dual_clip = None
        self._value_clip = True
        self._w_vf = 0.1
        self._w_ent = 0.01
        self.kl_coef = 0.5
        self._max_grad_norm = 100.0
        self.kl_target = 0.01
        self._vf_clip_para = 10.0
        
        
    def get_grpo_config(self):
        grpo_config = GRPOConfig(
            
            init_kl_coef=0.1,
            gradient_accumulation_steps=1,
            early_stopping=False,
            adap_kl_ctrl=True,
            grpo_epochs=1,
            gamma=0.99,
            lam=0.95,
            cliprange_value=0.2,
            vf_coef=0.01,  # 0.001
            target_kl=6,
            kl_penalty="kl",
            seed=123,
            
        )
        return grpo_config
    
   
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_critic_target.parameters(), self.actor_critic.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        
    def first_get_action(self,o_agent,m,o,u,f_0_list,pi_list,agent_ix):
        self.actor_critic.eval()
        
        # pi_agent = self.actor(o_agent,m)
        if self.label or self.volume:
            pi_agent,feature,label,_  = self.actor_critic(o_agent,m,o,u,agent_ix,f_0_list,pi_list,mark=1)
            pi_agent_ref,_,_,_  = self.actor_critic_ref(o_agent,m,o,u,agent_ix,f_0_list,pi_list,mark=1)
            return pi_agent,feature,pi_agent_ref,label
        else:
            pi_agent,feature ,_,value = self.actor_critic(o_agent,m,o,u,agent_ix,f_0_list,pi_list,mark=1)
            pi_agent_ref,_,_,_= self.actor_critic_ref(o_agent,m,o,u,agent_ix,f_0_list,pi_list,mark=1)
        
        return pi_agent,feature,pi_agent_ref
    

    def train(self, transitions,agent_ix,msg):
        self.actor_critic.train()
        self.pi_optimizer.zero_grad() 
        
        r = transitions['reward']
        o, u, o_next = [], [], []  
       
        o = transitions['o']
        u = transitions['u']
        o_next = transitions['o_next']
       
        u_next = []
        f_0_list = transitions['f_0']
        pi_list = transitions['pi']

        log = transitions['log']
        log_ref = transitions['log_ref']
         
        loss_grpoes = 0.0
        pi_agent_all = []
        pi_agent_target_all = []
         
        for time_step in range(self.max_time_step):
            o_agent = []
            u_agent = []
            o_next_agent = []
            log_agent = []
            r_agent = []
            for k in range(len(o)):
                o_agent.append(o[k][time_step][agent_ix].unsqueeze(0))
                u_agent.append(u[k][time_step][agent_ix].unsqueeze(0))
                o_next_agent.append(o_next[k][time_step][agent_ix].unsqueeze(0))
                log_agent.append(log[k][time_step][agent_ix].unsqueeze(0))
                r_agent.append(r[k][time_step][agent_ix].unsqueeze(0))
            o_agent = torch.cat(o_agent).float().to(self.device)
            u_agent = torch.cat(u_agent).float().to(self.device)
            o_next_agent = torch.cat(o_next_agent).float().to(self.device)
            log_agent = torch.cat(log_agent).float().to(self.device)
            r_agent = torch.cat(r_agent).float().to(self.device)
             
            m = msg[:,time_step,:] 
            if self.volume:

                pi_agent,feature,_,_ = self.actor_critic(o_agent,m,o,u,agent_ix,f_0_list,pi_list,0)
                
                pi_agent_target,_,_,_ = self.actor_critic_target(o_next_agent,m,o_next,u,agent_ix,f_0_list,pi_list,0)
                
                pi_agent_ref,_,_,_ = self.actor_critic_ref(o_agent,m,o,u,agent_ix,f_0_list,pi_list,mark=1)
            else:
                pi_agent,feature ,losses,_= self.actor_critic(o_agent,m,o,u_agent,agent_ix,f_0_list,pi_list,0)
            
                pi_agent_target,_ ,_,_ = self.actor_critic_target(o_next_agent,m,o_next,u_agent,agent_ix,f_0_list,pi_list,0)
            
                pi_agent_ref,_,_,_= self.actor_critic_ref(o_agent,m,o,u_agent,agent_ix,f_0_list,pi_list,mark=1)
             
             
            pi_agent_all.append(pi_agent)
            pi_agent_target_all.append(pi_agent_target)
            
            pg_loss,loss_grpo = self.grpo_trainer.loss_diffusion(log_agent,r_agent,pi_agent,pi_agent_ref)
            loss_grpoes += loss_grpo

            
        loss_grpoes = loss_grpoes/self.max_time_step
        
                            
        return pi_agent_all,pi_agent_target_all,\
            self.pi_optimizer,self.pi_scheduler,\
           loss_grpo
    
    
    def train_ppo_new(self, transitions,agent_ix,msg=None):
        self.actor_critic.train()
        
        r = transitions['reward']
        o = transitions['o']
        u = transitions['u']
        o_next = transitions['o_next']
        m = transitions['msg']
        f_0_list = transitions['f_0']
        pi_list = transitions['pi']
        done = transitions['done']
        if msg is not None:
            m = msg

        o_agent = []
        u_agent = []
        o_next_agent = []
        r_agent = []
        done_agent = []
        msg_agent = []
        for k in range(len(o)):
            o_ = []
            u_ = []
            o_next_ = []
            r_ = []
            done_ = []
            m_ = []
            for time_step in range(self.max_time_step):
                o_.append(o[k][time_step][agent_ix].unsqueeze(0))
                u_.append(u[k][time_step][agent_ix].unsqueeze(0))
                o_next_.append(o_next[k][time_step][agent_ix].unsqueeze(0))
                done_.append(done[k][time_step][agent_ix].unsqueeze(0))
                r_.append(r[k][time_step][agent_ix].unsqueeze(0))
                m_.append(m[k][time_step].unsqueeze(0))
            o_agent.append(torch.cat(o_).unsqueeze(0))
            u_agent.append(torch.cat(u_).unsqueeze(0))
            o_next_agent.append(torch.cat(o_next_).unsqueeze(0))
            r_agent.append(torch.cat(r_).unsqueeze(0))
            done_agent.append(torch.cat(done_).unsqueeze(0))
            msg_agent.append(torch.cat(m_).unsqueeze(0))
        o_agent = torch.cat(o_agent).float().to(self.device)
        u_agent = torch.cat(u_agent).float().to(self.device)
        o_next_agent = torch.cat(o_next_agent).float().to(self.device)
        r_agent = torch.cat(r_agent).float().to(self.device)
        done_agent = torch.cat(done_agent).float().to(self.device)
        msg_agent = torch.cat(msg_agent).float().to(self.device)
         
        
        r_agent = r_agent.reshape(-1)
        done_agent = done_agent.reshape(-1)
        
        o_next_agent = o_next_agent
        o_agent = o_agent
        
        
        with torch.no_grad():
            
            value_nexts = []
            for tt in range(o_next_agent.shape[1]):
                _,_,_,value_next = self.actor_critic(o_next_agent[:,tt,:],msg_agent[:,tt],o,\
                                                    u_agent[:,tt,:],agent_ix,f_0_list,pi_list,0)
                value_nexts.append(value_next.unsqueeze(0))
             
            value_nexts = torch.concat(value_nexts).permute(1,2,0,3).reshape(-1)
        
        returns = self.process_fn(r_agent.cpu(),done_agent.cpu(),o_next_agent.squeeze(0).cpu(),value_nexts.cpu())
         
        losses, clip_losses, vf_losses, ent_losses, kl_losses = [], [], [], [], []
        
        with torch.no_grad():
            act = []
            value = []
            for tt in range(o_agent.shape[1]):
                act_,_,_,value_ = self.actor_critic(o_agent[:,tt,:],msg_agent[:,tt],o,u_agent[:,tt,:],agent_ix,f_0_list,pi_list,0)
                act.append(act_.unsqueeze(0))
                value.append(value_.unsqueeze(0))
            act = torch.concat(act).permute(1,0,2,3)
            value = torch.concat(value).permute(1,0,2,3)
             
            old_logits = act
            if isinstance(act, tuple):
                dist = self.dist_fn(*act)
            else:
                dist = self.dist_fn(act)
            act_ = torch.argmax(act, dim=-1)
        v = value
        old_log_prob = dist.log_prob(to_torch_as(act_, v))
        logp_old = old_log_prob
        
        returns = to_torch_as(returns, v[0]).reshape(v.shape[0],v.shape[2],v.shape[1],v.shape[-1]).permute(0,2,1,3)
        adv = returns - v
         
        if self._rew_norm:
            mean, std = adv.mean(), adv.std()
            if not np.isclose(std.item(), 0):
                adv = (adv - mean) / std
        act = torch.argmax(u_agent,dim=-1)
        adv = adv.squeeze(-1)
        repeat = 5
        model_loss_time = [0.0]*8
        for _ in range(repeat):
            pi_agent = []
            
            value = []
            for tt in range(o_next_agent.shape[1]):
                pi_agent_,feature,model_loss_,value_ = self.actor_critic(o_agent[:,tt,:],msg_agent[:,tt],o,u_agent[:,tt,:],agent_ix,f_0_list,pi_list,0)
                pi_agent.append(pi_agent_.unsqueeze(0))
                model_loss_time[tt]+=model_loss_
                value.append(value_.unsqueeze(0)) 
            pi_agent = torch.concat(pi_agent).permute(1,0,2,3)
            
            value = torch.concat(value).permute(1,0,2,3)
            if isinstance(pi_agent, tuple):
                dist = self.dist_fn(*pi_agent)
            else:
                dist = self.dist_fn(pi_agent)
            dist_ = dist
            
            ratio = (dist_.log_prob(act) - logp_old).exp().float()
            
            surr1 = ratio * adv
            surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * adv
             
            if self._dual_clip:
                clip_loss = -torch.max(torch.min(surr1, surr2), self._dual_clip * adv).mean()
            else:
                clip_loss = -torch.min(surr1, surr2).mean()
            clip_losses.append(clip_loss.item())
            if self._value_clip:
                v_clip = v + (value - v).clamp(-self._vf_clip_para, self._vf_clip_para)
                vf1 = (returns - value).pow(2)
                vf2 = (returns - v_clip).pow(2)
                vf_loss = torch.max(vf1, vf2).mean()
            else:
                vf_loss = (returns - value).pow(2).mean()
             
            kl = torch.distributions.kl.kl_divergence(self.dist_fn(old_logits), dist_)
            kl_loss = kl.mean()
            kl_losses.append(kl_loss.item())
            vf_losses.append(vf_loss.item())
            e_loss = dist_.entropy().mean()
            ent_losses.append(e_loss.item())
           
            loss = clip_loss + self._w_vf * vf_loss - self._w_ent * e_loss + self.kl_coef * kl_loss # + model_loss
            
            losses.append(loss)
            self.pi_optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_critic.parameters()), self._max_grad_norm,
            )
            
            self.pi_optimizer.step()
            self.pi_scheduler.step()
            
         
        cur_kl = np.mean(kl_losses)
        if cur_kl > 2.0 * self.kl_target:
            self.kl_coef *= 1.5
        elif cur_kl < 0.5 * self.kl_target:
            self.kl_coef *= 0.5
        res = {
            "loss/total_loss": losses,
            "loss/policy": clip_losses,
            "loss/vf": vf_losses,
            "loss/entropy": ent_losses,
            "loss/kl": kl_losses,
        }
        
        total = sum(res['loss/total_loss'])/repeat
         
        return res
    
    def compute_episodic_return(
        self,
        reward,
        done,
        v_s_ = None,
        gamma = 0.99,
        gae_lambda = 0.95,
        rew_norm = False,
    ):
        """Compute returns over given full-length episodes.
        Implementation of Generalized Advantage Estimator (arXiv:1506.02438).
        :param batch: a data batch which contains several full-episode data
            chronologically.
        :type batch: :class:`~tianshou.data.Batch`
        :param v_s_: the value function of all next states :math:`V(s')`.
        :type v_s_: numpy.ndarray
        :param float gamma: the discount factor, should be in [0, 1], defaults
            to 0.99.
        :param float gae_lambda: the parameter for Generalized Advantage
            Estimation, should be in [0, 1], defaults to 0.95.
        :param bool rew_norm: normalize the reward to Normal(0, 1), defaults
            to False.
        :return: a Batch. The result will be stored in batch.returns as a numpy
            array with shape (bsz, ).
        """
        rew = reward
        v_s_ = np.zeros_like(rew) if v_s_ is None else to_numpy(v_s_.flatten())
        
        returns = _episodic_return(v_s_, rew, done, gamma, gae_lambda)
       
        if rew_norm and not np.isclose(returns.std(), 0.0, 1e-2):
            returns = (returns - returns.mean()) / returns.std()
        
        return returns

    
    
    def process_fn(self, rew,done,obs_next,value):
        
        
         
        if self._rew_norm:
            mean, std = rew.mean(), rew.std()
            if not np.isclose(std, 0):
                rew = (rew - mean) / std
        
        if self._lambda in [0, 1]:
            
            return self.compute_episodic_return(rew,done, None, 1.0, self._lambda)
        else:
            
            v_ = []
           
            v_ = value
            v_ = to_numpy(v_)
            
           
            return self.compute_episodic_return(rew,done, v_, 1.0, self._lambda)

    def save_model(self, train_step,ix,model_path):
        self.save_rate = 20
        num = str(train_step // self.save_rate)
         
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        torch.save(self.actor_critic.state_dict(), model_path + '/' + num +'_'+str(ix)+ '_MGN_params.pkl')
        

    def save_best_model(self,train_step,i,model_path):
        self.save_rate = 20
        num = str(train_step // self.save_rate)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_critic.state_dict(), model_path + '/'  +str(i)+ 'best_MGN_params.pkl')
        

    def load_best_model(self,ix,model_path):
        
        self.actor_critic.load_state_dict(torch.load(model_path + '/' +str(ix)+ 'best_MGN_params.pkl'))
        

    
    def print_grad(self):
       
        print("actor")
        for name, parms in self.actor_critic.named_parameters():	
            print('-->name:', name)
            print('-->para:', parms)
            print('-->grad_requirs:',parms.requires_grad)
            print('-->grad_value:',parms.grad)
            print("===")
            break