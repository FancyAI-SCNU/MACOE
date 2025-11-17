from runner_am_ind_mla_gru import Runner
from common.arguments import get_args
import numpy as np
import random
import torch

np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)

from env_rl_ma_ind import StockEnv
 
if __name__ == '__main__':
    
    args = get_args()
    
    env = StockEnv(args)
    
    runner = Runner(args, env)
     
    runner.run()
