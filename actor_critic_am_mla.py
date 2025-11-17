import numpy as np
import torch
from torch import nn
from itertools import chain
import math
EPS = 1e-8

np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)

class MLP(nn.Module):
    def __init__(
        self,
        shape,
        hidden_sizes=(32,), 
        activation="tanh", 
        output_activation=None, 
        bn=False, 
        is_training=np.bool_(0)
    ) :
        super(MLP, self).__init__()
        self.hidden = hidden_sizes[:-1]
        
        self.mlp_layers=[]
        ix = 0
        for h in self.hidden:
            if bn:
                if ix==0:
                    self.mlp_layers.append(nn.Linear(shape,h))
                else:
                    self.mlp_layers.append(nn.Linear(self.hidden[ix-1],h))
                self.mlp_layers.append(nn.BatchNorm1d(h))
                self.mlp_layers.append(nn.ReLU())
            else:
                if ix==0:
                    self.mlp_layers.append(nn.Linear(shape,h))
                else:
                    self.mlp_layers.append(nn.Linear(self.hidden[ix-1],h))
                
                act = nn.Tanh() if activation=="tanh" else nn.ReLU()
                self.mlp_layers.append(act)
            ix +=1
        if len(self.mlp_layers)!=0:
            self.mlp_layers.append(nn.Linear(hidden_sizes[-2],hidden_sizes[-1]))
        else:
            self.mlp_layers.append(nn.Linear(shape,hidden_sizes[-1]))
        if output_activation is not None:
            act_out = nn.Tanh() if activation=="tanh" else nn.ReLU()
            self.mlp_layers.append(act_out)
        
        self.mlp = nn.ModuleList(self.mlp_layers
                )
        
    def forward(self,x):
        for layer in self.mlp:
            x = layer(x)
        return x
        

class MLP_Model(nn.Module):
    def __init__(
        self,
        shape,
        hidden_sizes=(32,), 
        activation="tanh", 
        output_activation=None, 
        bn=False, 
        is_training=np.bool_(0)
    ) :
        super(MLP_Model, self).__init__()
        self.hidden = hidden_sizes[:-1]
        self.mlp_layers=[]
        ix = 0
        
        for h in self.hidden:
            
            if ix==0:
                self.mlp_layers.append(nn.Linear(shape,h))
            else:
                self.mlp_layers.append(nn.Linear(self.hidden[ix-1],h))
            
            self.mlp_layers.append(nn.ReLU())
            ix +=1
        if len(self.mlp_layers)!=0:
            self.mlp_layers.append(nn.Linear(hidden_sizes[-2],hidden_sizes[-1]))
        else:
            self.mlp_layers.append(nn.Linear(shape,hidden_sizes[-1]))
        if output_activation is not None:
            act_out = nn.Tanh() if activation=="tanh" else nn.ReLU()
            self.mlp_layers.append(act_out)
        
        self.mlp = nn.ModuleList(self.mlp_layers
                )
        

    def forward(self,x):
        
        for layer in self.mlp:
            
            x = layer(x)
           
        return x
    

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
        self.act_1 = nn.Tanh() if activation=="tanh" else nn.ReLU()
        self.layer_3 = nn.Linear(hidden_sizes[0]*2,hidden_sizes[1])
        self.act_2 = nn.Tanh() if activation=="tanh" else nn.ReLU()
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
    

        
class Q_fun(nn.Module):
    def __init__(
        self,
        n_agents, 
        x, 
        a, 
        activation="relu",
        hidden_sizes=(300,300),
        
    ) :
        super(Q_fun, self).__init__()
        obs_dim = x
        act_dim = a
        self.n_agents = n_agents

        shape_cen_q_1 = obs_dim + act_dim + 1
        shape_cen_q_2 = obs_dim*(n_agents) + act_dim*(n_agents)
       
        self.mlp_cen_q = MLP_Cen(shape_cen_q_1,shape_cen_q_2,list(hidden_sizes)+[1],activation)

    def forward(self,x,a,pi_,pi_other,agent):
        
        q_input_1 = torch.concat([x[agent,:,:],a[agent,:,:]],dim=-1)
        
        q_input_2 = torch.concat([x,a],dim=-1)
        id_mark = torch.zeros([q_input_1.shape[0],1]).to(q_input_1.device)
        id_mark += agent/self.n_agents 
         
        q_input_2 = q_input_2.reshape(q_input_1.size(0),q_input_1.size(1)*(self.n_agents))

        q_input_1 = torch.concat([q_input_1,id_mark],dim=-1)
        
        q_ = self.mlp_cen_q([q_input_1,q_input_2])
        if q_.isnan().any():
            print(q_input_1.isnan().any(),"1111")
            print(q_input_2.isnan().any(),"2222")
        
        q_pi_input_1 = torch.concat([x[agent,:,:],pi_],dim=-1)
         
        q_pi_input_2 = torch.concat([x,torch.tensor(pi_other).permute(1,0,2)],dim=-1)
        q_pi_input_2 = q_pi_input_2.reshape(q_pi_input_1.size(0),q_pi_input_1.size(1)*(self.n_agents))
        
        q_pi_input_1 = torch.concat([q_pi_input_1,id_mark],dim=-1)
        q_pi_ = self.mlp_cen_q([q_pi_input_1,q_pi_input_2])
        
        return q_,q_pi_

