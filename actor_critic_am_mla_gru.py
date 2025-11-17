import numpy as np
import torch
from torch import nn
from itertools import chain
import math
EPS = 1e-8

np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)


    

class MLP_Cen(nn.Module):
    def __init__(
        self,
        shape1,
        shape2,
        hidden_sizes=(32,), 
        activation="tanh", 
        output_activation=None, 
    ) :
        super(MLP_Cen, self).__init__()
        
        
        self.layer_1 = nn.Linear(shape1,hidden_sizes[0])
        self.act = nn.Tanh() if activation=="tanh" else nn.ReLU()
        self.layer_2 = nn.Linear(shape2,hidden_sizes[0])
        
        self.layer_3 = nn.Linear(hidden_sizes[0]*2,hidden_sizes[1])
        self.layer_4 = nn.Linear(hidden_sizes[1],hidden_sizes[-1])
        if output_activation is not None:
            self.act_out = nn.Tanh() if activation=="tanh" else nn.ReLU()
        else:
            self.act_out =None

    def forward(self,xx):
        x_ = xx[0]
        other_ = xx[1]
        
        x = self.act(self.layer_1(x_))
        
        other = self.act(self.layer_2(other_))
        
        
        x = self.act(self.layer_3(torch.concat([x,other],-1)))
        x = self.layer_4(x)
        if  self.act_out is not None:
            x = self.act_out(x)
        
        return x
    

class actor_critic(nn.Module):
    def __init__(
        self,
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
        max_est_time=5, msg_idx=5
    ) :
        super(actor_critic, self).__init__()
        
       
        act_dim = 5
        obs_dim = 78*2
        self.act_limit = 1
        msg_dim = 80 
        att_dim = 5
        om_msg_dim = msg_dim

        self.msg_dim = msg_dim
        self.n_agents = n_agents
        
        self.pi, self.q, self.q_pi, self.msg = [], [], [], []
        
        self.cnn_shape = [30,6]
        hidden_size = 64

        shape_pi = obs_dim+msg_dim
        
        self.agent_stocks=10

        self.msg_idx = 5

        self.om_act_all, om_msg_all = [], []
       
        
        self.cnn = nn.Sequential(nn.Conv1d(self.cnn_shape[1], 3, 3), nn.ReLU())
        self.raw_fc = nn.Sequential(nn.Linear((self.cnn_shape[0] - 2) * 3, 64), nn.ReLU())

        self.rnn = nn.GRU(64, hidden_size, batch_first=True)
        
        self.dnn = nn.Sequential(nn.Linear(18, 64), nn.ReLU())
        self.rnn2 = nn.GRU(64, hidden_size, batch_first=True)
        
        self.drop=nn.Dropout(0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 32), nn.ReLU()
        )

        
        self.fc_out = nn.Sequential(
            nn.Linear(32, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 32), nn.ReLU(),
        )
        
        self.layer_out = nn.Sequential(nn.Linear(32, act_dim), nn.Softmax(dim=-1))

        
        self.value_out = MLP_Cen(32,act_dim,list(hidden_sizes)+[1],activation)

        
        self.label = False
        self.volume = False
        
        self.linear_mlp_pedict = nn.Linear(shape_pi,hidden_sizes[0])
        self.linear_mlp_pedict_2 = nn.Linear(hidden_sizes[0],30)
        self.linear_mlp_pedict_short = nn.Linear(hidden_sizes[0],8)
        self.linear_mlp_pedict_long = nn.Linear(hidden_sizes[0],8)

        
    def pi_start(self,x_agent,m):
         
        if len(x_agent.shape)<3:
            x_agent = x_agent.unsqueeze(0)
        
        agent_stocks = x_agent.shape[-2]
        x_agent = x_agent.reshape(-1,x_agent.shape[-2],x_agent.shape[-1])
        batch_num = x_agent.shape[0]

        m = m.reshape(batch_num,-1,80)
        m_ = m.repeat(1,agent_stocks, 1)
        
        
        x_1 = x_agent[:,:,:60]
        x_2 = x_agent[:,:,60:78]
        
        input = torch.concat([x_1,m_],dim=-1)
        
        input = torch.cat((torch.zeros_like(input[:,:, :40]), input), dim=-1)
        
        raw_in = input.reshape(batch_num*agent_stocks, 30, 6).transpose(1, 2)
        cnn_out = self.cnn(raw_in).view(batch_num, agent_stocks, -1)
        
        rnn_in = self.raw_fc(cnn_out)
        rnn_out = self.rnn(rnn_in)[0]
        
        rnn2_in = self.dnn(x_2)
        rnn2_out = self.rnn2(rnn2_in)[0]
       
        fc_in = torch.cat((rnn_out, rnn2_out), dim=-1)
        feature = self.fc(fc_in)
         
        pi_ = self.layer_out(feature)
       
        pi_ = pi_.reshape(batch_num,agent_stocks,-1)

        feature = self.fc_out(feature).reshape(batch_num,agent_stocks,-1)
        
        return pi_,feature
    

    
    
    def forward(self,x_agent,m,x,a,agent,f_0_output_list,pi_list,mark=0):
   
        losses = 0.0
        if len(m.shape)>3:
            m= m.squeeze(0)
        if len(x_agent.shape)>2:
            x_agent = x_agent.squeeze(0)
        
        pi_list = []
       
        pi,feature = self.pi_start(x_agent,m)
        
        pi_list.append(pi)
        
        value =  self.value_out([feature,pi])
        losses-=torch.mean(value)
        
        return pi,feature,losses,value
    
     
