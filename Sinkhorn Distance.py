import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ot_utils import *
from utils import *

import argparse
import time
# import gym
import torch

# from models.mlp_critic import Value
# from models.mlp_policy import Policy
from models.ot_critic import OTCritic

import pickle
import random
import numpy as np

parser = argparse.ArgumentParser(description='Sinkhorn Imitation Learning')
# parser.add_argument('--env-name',                    default="Ant-v2", metavar='G', help='name of the environment to run')
# parser.add_argument('--expert-traj-path',                                      metavar='G', help='path of the expert trajectories')
# parser.add_argument('--render', action='store_true', default=False,                         help='render the environment')
# parser.add_argument('--log-std', type=float,         default=-0.0,             metavar='G', help='log std for the policy ')
# parser.add_argument('--gamma', type=float,           default=0.99,             metavar='G', help='discount factor')
# parser.add_argument('--tau', type=float,             default=0.99,             metavar='G', help='gae')
# parser.add_argument('--l2-reg', type=float,          default=1e-3,             metavar='G', help='l2 regularization regression')
# parser.add_argument('--learning-rate', type=float,   default=3e-4,             metavar='G', help='lr')
# parser.add_argument('--num-threads', type=int,       default=4,                metavar='N', help='Threads')
parser.add_argument('--seed', type=int,              default=1,                metavar='N', help='Seed')
# parser.add_argument('--min-batch-size', type=int,    default=50000,             metavar='N', help='minimal batch size per TRPO update')
# parser.add_argument('--max-iter-num', type=int,      default=250,              metavar='N', help='maximal number of main iterations')
# parser.add_argument('--log-interval', type=int,      default=1,                metavar='N', help='interval between training status logs ')
# parser.add_argument('--save-model-interval',type=int,default=10,               metavar='N', help='interval between saving model ')
# parser.add_argument('--gpu-index', type=int,         default=0,                metavar='N', help='Index num of GPU to use')
# parser.add_argument('--max-kl', type=float,          default=0.1,             metavar='G', help='max kl value ')
# parser.add_argument('--damping', type=float,         default=0.1,             metavar='G', help='damping')
# parser.add_argument('--expert-samples', type=int,    default=1000,             metavar='G', help='expert sample number ')
# parser.add_argument('--wasserstein-p', type=int,     default=1,                metavar='G', help='p value for Wasserstein')
# parser.add_argument('--resume-training',             type=tools.str2bool,      nargs='?', const=True, default=False,  help='Resume training ?')
# parser.add_argument('--critic-lr', type=float,       default=5e-4,             metavar='G', help='Critic learning rate')
# parser.add_argument('--log-actual-sinkhorn',         type=tools.str2bool,      nargs='?', const=True, default=False,  help='Track actual Sinkhorn with normal cosine cost (for eval only)')
# parser.add_argument('--dataset-size', type=int,      default=4,                metavar='G', help='Number of trajectories')
# parser.add_argument('--use-mean',         type=tools.str2bool,      nargs='?', const=True, default=False,  help='how to sample from policy')


# python imitation-learning/SIL.py --env-name Ant-v2 --expert-traj-path assets/subsampled_expert_traj/16/Ant-v2 --gamma 0.99
# --tau 0.97 --min-batch-size 50000 --seed 123 --max-iter-num 250 --log-actual-sinkhorn True --critic-lr .0005 --dataset-size 16

args = parser.parse_args()
dtype = torch.float64
# torch.set_default_dtype(dtype)
device = torch.device('cpu')
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# if torch.cuda.is_available():
#     torch.cuda.set_device(args.gpu_index)
#     env   state_dim   is_disc_action   action_dim
# env = gym.make(args.env_name)
# state_dim      = env.observation_space.shape[0]
# is_disc_action = len(env.action_space.shape) == 0
# action_dim     = 1 if is_disc_action else env.action_space.shape[0]
state_dim      = 21
is_disc_action = 0
action_dim     = 21
action_space_dim = 1
resume_training = False
critic_lr = 5e-4
log_actual_sinkhorn = True
dataset_size = 4

running_reward  = ZFilter((1,), demean=False, clip=10)
print("Seed: {}".format(args.seed))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# env.seed     policy_net
# env.seed(args.seed)
# policy_net = Policy(state_dim, action_space_dim, log_std=args.log_std)
# value_net         = Value(state_dim)
critic_net        = OTCritic(state_dim + action_dim) # Adversarially learned critic

if resume_training:
    critic_net, running_reward = pickle.load(open('assets/learned_models/SIL/SIL_s{}.p'.format(args.seed), "rb"))

to_device(device, critic_net)

optimizer_ot      = torch.optim.Adam(critic_net.parameters(), lr=critic_lr)
# optimizer_value   = torch.optim.Adam(value_net.parameters(),  lr=args.learning_rate)
#optimizer_ot      = torch.optim.RMSprop(critic_net.parameters(), lr = args.critic_lr)
#OT params

epsilon           = 0.01                  # Entropy regularisation for Optimal Transport
niter             = 1000000000            # Max Sinknhorn iterations

# load trajectory
# expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))
# running_state.fix          = True


with open('data/插值obsqpos.json') as f:
    traj_data = json.load(f)
expert_traj = np.array([traj_data]) # todo json 专家轨迹 读取
expert_traj = np.concatenate([expert_traj[:, :, :21], expert_traj[:, :, -21:]], axis=2) # 只用qpos
print('expert_traj shape:', expert_traj.shape)

batch_obs_qpos = []  # todo json to batch data
for i in range(50):
    print('read data/itr_2240_skill_obs_qpos_%02d.json' % i)
    with open('data/itr_2240_skill_obs_qpos_%02d.json' % i) as f:
        traj_data = json.load(f)
        batch_obs_qpos.append(traj_data)
batch_obs_qpos = np.array(batch_obs_qpos)
batch_obs_qpos = np.concatenate([batch_obs_qpos[:, :, :21], batch_obs_qpos[:, :, -21:]], axis=2) # 只用qpos
print('batch_obs_qpos shape:', batch_obs_qpos.shape)

traj                       = []
idx                        = 0
dataset                    = []
offset                     = 0
# print("Dataset dimensions:", expert_traj.shape)
def sil_step(batch_obs_qpos):
    # states      = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    # actions     = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    to_device(device, critic_net)
    # X = torch.cat([states, actions], 1).to(dtype).to(device)  # Concatenate s,a pairs of agent
    X = torch.from_numpy(batch_obs_qpos).to(dtype).to(device)  # Concatenate s,a pairs of agent
    for _ in range(1):
        sampled_episodes = []
        for pair in range(len(X)):  # Split to episodes to do random matching (Corresponds to Step 3 of Algorithm 1)
            sampled_episodes.append(X[pair].cpu().numpy())
        total_wasserstein = 0  # Keeps track of all Wassersteins for one episode
        rewards = []  # Logs rewards to update TRPO
        min_wasserstein = 10e10  # Used for logging at command line
        max_wasserstein = 0  # Used for logging at command line
        best_trajectory = None  # Used for logging at command line
        worst_trajectory = None  # Used for logging at command line
        index = 0  # Used for logging at command line
        best_idx = 0  # Used for logging at command line
        worst_idx = 0  # Used for logging at command line
        per_trajectory_dis = []  # Used for logging at command line
        cost_loss = []
        num_of_samples = len(sampled_episodes) - 1
        # threshold = num_of_samples - 3
        episodic_eval_sinkhorn = []
        for trajectory in sampled_episodes:
            X = torch.tensor(np.array(trajectory)).to(dtype).to(device)  # Convert trajectory to tensor.
            sample_traj_index = random.randint(0, (len(expert_traj) - 1))
            Y = torch.from_numpy(expert_traj[sample_traj_index]).to(dtype).to(
                device)  # Randomly match (Corresponds to Step 3 of Algorithm 1)
            cost_matrix = cosine_critic(X, Y, critic_net)  # Get cost matrix for samples using critic network.
            transport_plan = optimal_transport_plan(X, Y, cost_matrix, method='sinkhorn')  # Getting optimal coupling
            per_sample_costs = torch.diag(torch.mm(transport_plan,
                                                   cost_matrix.T))  # Get diagonals W = MC^T, where M is the optimal transport map and C the cost matrix
            distance = torch.sum(per_sample_costs)  # Calculate Wasserstein by summing diagonals, i.e., W=Trace[MC^T]
            wasserstein_distance = -(
                distance)  # Assign -wasserstein in order to GD to maximise if using adversary for training.

            per_trajectory_dis.append(
                distance.detach().cpu().numpy())  # Keep track of all Wasserstein distances in one sample.

            # =========FOR EVALUATION ONLY=============#
            if log_actual_sinkhorn:
                evaluation_cost_matrix = cosine_distance(X, Y)
                evaluation_transport_plan = optimal_transport_plan(X, Y, evaluation_cost_matrix, method='sinkhorn')
                eval_wasserstein_distance = torch.sum(
                    torch.diag(torch.mm(evaluation_transport_plan, evaluation_cost_matrix.T)))
                episodic_eval_sinkhorn.append(eval_wasserstein_distance.item())
            # =========================================#

            if distance < min_wasserstein and index != (
            len(sampled_episodes)):  # Keep track of best trajectory based on Wasserstein distance
                min_wasserstein = distance
                best_trajectory = X
                best_idx = index
            if distance > max_wasserstein and index != (
            len(sampled_episodes)):  # Keep track of worst trajectory based on Wasserstein distance
                max_wasserstein = distance
                worst_trajectory = X
                worst_idx = index
            index += 1
            counter = 0
            survival_bonus = 4 / X.shape[0]
            for per_sample_cost in per_sample_costs:
                with torch.no_grad():
                    temp_r = -2 * per_sample_cost + survival_bonus
                    temp_r.unsqueeze_(0)
                    temp_r = running_reward(temp_r.cpu())
                    rewards.append(temp_r)
                    counter += 1
            total_wasserstein += distance
            torch.cuda.empty_cache()
    total_wasserstein = -total_wasserstein / num_of_samples
    optimizer_ot.zero_grad()
    total_wasserstein.backward()  # Only backpropagates through the critic network.
    optimizer_ot.step()
    #    args.critic_lr*=0.992 # Perhaps decreasing lr may be useful in stabilizing training process
    #    for param_group in optimizer_ot.param_groups:
    #        param_group['lr'] = args.critic_lr
    # with torch.no_grad():
    #     rewards = torch.tensor(rewards)
    torch.cuda.empty_cache()
    return (total_wasserstein ** 2) ** (1 / 2), episodic_eval_sinkhorn, len(
        sampled_episodes), min_wasserstein, best_trajectory, best_idx, max_wasserstein, worst_trajectory, worst_idx, per_trajectory_dis


def sil():
    max_iter_num = 250
    save_model_interval = 10
    print("------------ Sinkhorn Imitation Learning (SIL) -------------")
    print("---Parameters:----\n")
    # print("KL: {}"                           .format(args.max_kl))
    # print("Damping: {}"                      .format(args.damping))
    # print("Value Function Regularisation: {}".format(args.l2_reg))
    # print("Critic Learning Rate: {}"         .format(args.critic_lr))
    # print("γ: {}"                            .format(args.gamma))
    # print("τ: {}"                            .format(args.tau))
    W_loss = []
    sinkhorn_log = []
    if resume_training == True:
        sinkhorn_log = pickle.load(open('experiment-logs-sil/skh_seed{}.l'.format(args.seed), "rb"))
        input("continue ?")
    episode = []
    loss = 10e3
    for i_iter in range(max_iter_num):
        t0 = time.time()
        loss, eval_sinkhorn_per_episode, sampled_episodes, min_loss, best_trajectory, best_idx, max_loss, worst_trajectory, worst_idx, per_trajectory_dis = sil_step(
            batch_obs_qpos)
        sinkhorn_log.append(eval_sinkhorn_per_episode)
        episode.append(i_iter)
        W_loss.append(eval_sinkhorn_per_episode)
        if save_model_interval > 0 and (i_iter) % save_model_interval == 0:
            to_device(torch.device('cpu'), critic_net)
            pickle.dump((critic_net, running_reward), open(os.path.join(assets_dir(), 'learned_models/SIL/SIL_s{}.p'.format(args.seed)), 'wb'))
            pickle.dump(sinkhorn_log, open('experiment-logs-sil/skh_seed{}.l'.format(args.seed), 'wb'))
        torch.cuda.empty_cache()
        t1 = time.time()
        print('iter {}/{}\tAdv. Sinkhorn {:.4f}\tActual Sinkhorn {:.4}\tSampled Episodes {}'.format(i_iter,max_iter_num,loss.item(),
                                                                                          sum(eval_sinkhorn_per_episode) / sampled_episodes,
                                                                                          sampled_episodes))


sil()
