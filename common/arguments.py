import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")
     
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=8e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=3e-4, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=16, help="number of episodes to optimize at the same time")
    parser.add_argument("--epoch", type=int, default=500, help="number of episodes to optimize at the same time")
    
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")
    # DATA parameters
    parser.add_argument("--n_agents", type=int, default=8, help="number of agents")
    parser.add_argument("--attn_dim", type=int, default=5, help="dim of attn")
    parser.add_argument("--msg_dim", type=int, default=80, help="dim of message")
    parser.add_argument("--time_steps", type=int, default=3, help="len of the imaged")
    # gpu
    parser.add_argument('--gpu', default='6', type=str,help='id(s) for CUDA_VISIBLE_DEVICES')
    # env
    parser.add_argument("--max_step_num", type=int, default=8, help="step number")
    parser.add_argument("--limit", type=int, default=10, help="limits")
    # change
    parser.add_argument("--time_interval", type=int, default=30, help="time_interval")
    parser.add_argument("--interval_num", type=int, default=8, help="limits")
    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()

    return args
