import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import copy
import dgl
import torch.optim as optim
from RL_Algorithm.model.net import Actor
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from RL_Algorithm.utils.rl_utils import edge_mask,virtual_node_g,graph_batch
import pickle
from RL_Algorithm.model.generate_pretrain_data import low_degree_generate

def create_tensor(position,length):
    tensor = torch.zeros(length)  # 创建一个长度为10的全零张量
    tensor[position] = 1  # 将指定位置的值设为1
    return tensor
def pretrain_actor_net(args,data_size=100,path=''):
    graph_type = args.graph_type
    n = args.num_nodes
    # 预训练数据
    try:
        with open(path+'Pretrain_data_%s_%s_%s_low_degree_design.pickle'% (data_size,graph_type,n), 'rb') as file:
            pre_train_data = pickle.load(file)
    except:
        pre_train_data = low_degree_generate(data_size,graph_type,n,path =path)
    G_list = [x[1][0:-1] for x in pre_train_data]
    action_list = [x[2] for x in pre_train_data]
    all_G_list = [y for x in G_list for y in x]
    all_action_list = [y for x in action_list for y in x]
    # 预训练数据处理
    inverse_action_dict = dict()  # 边对应的action index  # 1:一步生成边序号  2:分两步依次生成两个节点   3:一步生成两个节点
    e = 0
    for i in range(0, args.num_nodes - 1):
        for j in range(i + 1, args.num_nodes):
            inverse_action_dict[(i,j)] = e
            e += 1
    action_space = len(inverse_action_dict)
    states = [virtual_node_g(G) for G in all_G_list]
    edge_mask_matrixes = [edge_mask(G) for G in all_G_list]
    action_experts = [inverse_action_dict[tuple(sorted(action))] for action in all_action_list]
    action_experts_tensor =torch.tensor(action_experts)
    torch.save([states,edge_mask_matrixes,action_experts_tensor],path+'Pretrain_tensor_%s_%s_%s.pth'%(data_size,graph_type,n))
    states,edge_mask_matrixes,action_experts_tensor = torch.load(path+'Pretrain_tensor_%s_%s_%s.pth'%(data_size,graph_type,n))

    # 模型
    actor_net = Actor(args).to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=args.lr_a, eps=1e-5)
    num_epochs = 10000
    ep_loss_list = []
    best_net = copy.deepcopy(actor_net.state_dict())
    best_loss = 9999
    for epoch in range(num_epochs):
        ep_loss = 0
        # for index in BatchSampler(SubsetRandomSampler(range(len(pre_train_data))), args.mini_batch_size, False):
        for index in BatchSampler(SubsetRandomSampler(range(len(states))), args.mini_batch_size, False):
            batch_state, batch_state_features, batch_edge_mask_matrixes = graph_batch([states[ind] for ind in index],
                                                                                      [edge_mask_matrixes[ind] for ind in
                                                                                       index])
            action_probs = actor_net(batch_state.to(device), batch_state_features.to(device),batch_edge_mask_matrixes,pre_train=True)
            batch_action_expert =torch.stack([action_experts_tensor[ind] for ind in index]).to(device)
            actor_optimizer.zero_grad()
            loss = criterion(action_probs, batch_action_expert)
            ep_loss +=loss.item()*len(index) #去梯度
            loss.backward()
            actor_optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {ep_loss:.4f}')
        if ep_loss<best_loss:
            best_net = copy.deepcopy(actor_net.state_dict())
            best_loss = ep_loss
        if epoch%100==0:
            torch.save([best_net,args],path+'Pretrain_policy_%s_%s_%s.pth'%(data_size,graph_type,n))
        ep_loss_list.append(ep_loss/5000)
        ep_loss_list.append(ep_loss/5000)
    plt.figure()
    plt.plot(ep_loss_list)
    plt.savefig(path+'Pretrain_loss_%s_%s_%s.jpg'%(data_size,graph_type,n))
    plt.show()
    torch.save([best_net,args],path+'Pretrain_policy_%s_%s_%s.pth'%(data_size,graph_type,n))
    return best_net



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO of graph generation")
    # 环境参数
    parser.add_argument("--graph_type", type=str, default='BA', help="ER,BA,zero,Tree,WS0.3,WS0.1,WS0.5")
    parser.add_argument("--dismantling_name", type=str, default='GNDR', help="RB,RD,CI2,MS,GND,GNDR")
    parser.add_argument("--dismantling_number", type=int, default=int(4), help="瓦解目标size")
    parser.add_argument("--train_seed",  default=[100,150])
    parser.add_argument("--train_graph_ids", type=list, default=[100,101,102,103,104])
    parser.add_argument("--test_graph_ids", type=list, default=[150,151,152,153,154])
    parser.add_argument("--reward_type", type=str, default="slope", help="slope,area1,area2,area3")

    parser.add_argument("--num_nodes", type=int, default=int(100), help=" The number of nodes in the network")
    parser.add_argument("--pretrain", type=bool, default=False, help="Generality of training")
    parser.add_argument("--env_penalty", type=float, default=0, help=" Penalty for not adding new edge")

    # PPO参数
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor") #0.9
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    # parser.add_argument("--max_episode_steps", type=int, default=int(1600), help=" Maximum number of training steps")
    parser.add_argument("--max_episode_steps", type=int, default=int(40000), help=" Maximum number of training steps")
    parser.add_argument("--value_first", type=int, default=200, help="Train value network x step")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size") #5000
    parser.add_argument("--mini_batch_size", type=int, default=500, help="Minibatch size") # 500
    parser.add_argument("--update_time", type=int, default=10, help="PPO parameter")
    parser.add_argument("--lr_a", type=float, default=0.0002, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=0.0001, help="Learning rate of critic")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--grad_norm", type=float, default=0.5, help="max_grad_norm")
    # 评估频率
    parser.add_argument("--evaluate_freq", type=float, default=1, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--show_evaluate_freq", type=float, default=1, help="Evaluate the policy every 'evaluate_freq' steps")
    # parser.add_argument("--save_freq", type=int, default=200, help="Save frequency")
    # network 参数
    parser.add_argument("--dropout", type=str, default=0.0, help="dropout")
    parser.add_argument("--node_initial", type=str, default='structure', help="初始节点特征eye,one,structure")
    parser.add_argument("--feat_dim", type=int, default=int(7), help="初始节点特征维度50/9")
    parser.add_argument("--gcn_activation", type=str, default='relu', help="activation function of networ: relu, elu, tanh")
    parser.add_argument("--re_activation", type=str, default='relu', help="activation function of networ: relu, elu, tanh")
    parser.add_argument("--critic_re_activation", type=str, default='leakyrelu', help="activation function of networ: relu, elu, tanh")
    parser.add_argument("--node_embed_norm2", type=bool, default=True, help="是否对node embedding结果进行归一化")
    parser.add_argument("--graph_embed", type=str, default='virtual', help="Graph embedding methods: mean,sum")   # False
    parser.add_argument("--edge_embed", type=str, default='minus_add', help="Graph embedding methods: concat,minus_add")  # minus_add
    parser.add_argument("--hidden_dim1", type=int, default=int(64),help="The number of neurons in hidden layers of the neural network")  # 64
    parser.add_argument("--hidden_dim2", type=int, default=int(16), help="The number of neurons in hidden layers of the neural network") # 64
    parser.add_argument("--num_layers", type=int, default=int(3), help="The number of gcn layers")
    parser.add_argument("--gnn_type", type=str, default='GATv2', help="GraphSage,GATv2,GCN")  # GCN
    parser.add_argument("--aggregator_type", type=str, default='gcn', help="GraphSage: mean,lstm,gcn,pool")
    parser.add_argument("--num_heads", type=int, default=int(8), help="GATv2: the number of attention heads")
    parser.add_argument("--norm", type=str, default='none', help="GCN: norm by degree both or none")  # both

    args = parser.parse_args()
    pretrain_actor_net(args)