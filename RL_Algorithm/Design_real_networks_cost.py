import torch
import numpy as np
from RL_Algorithm.Environment.dismantlers.dismantlers_ import dismantle
from RL_Algorithm.model.PPO_cost import PPO_cost
from RL_Algorithm.model.local_search import local_search_cost
import copy
from itertools import count
import pickle
import os
import networkx as nx
from RL_Algorithm.utils.rl_utils import edge_mask_cost,virtual_node_g,virtual_node_g_cost,action_to_edge, create_edge_indices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_policy(G0,agent,node_cost,sum_cost,attack):
    target_size = 1 if attack not in ['GND', 'GNDR'] else 4
    G = copy.deepcopy(G0)
    now_cost = sum_cost
    edge_mask_matrix, edge_cost = edge_mask_cost(G0, now_cost, node_cost)
    G_list = [G0]
    edge_list = []
    for t in count():
        if agent.actor_net.fc1.in_features==7:
            state = virtual_node_g(G)
        if agent.actor_net.fc1.in_features == 8:
            state = virtual_node_g_cost(G, node_cost, now_cost)
        action, action_prob = agent.select_action(state, edge_mask_matrix, edge_cost, evaluation=True)
        act = action_to_edge(action, len(G0))
        now_cost -= (node_cost[act[0]] + node_cost[act[1]])
        G.add_edge(act[0], act[1])
        G_list.append(copy.deepcopy(G))
        edge_list.append(act)
        edge_mask_matrix, edge_cost = edge_mask_cost(G, now_cost, node_cost)
        if not edge_mask_matrix.any():  # 其中没有True
            episode_reward,curve,_ = dismantle(attack,G,target_size,agent.path+"/")
            break
    R_G=[episode_reward,curve,copy.deepcopy(G),G_list,edge_list]
    return R_G

def main(G0,real_name,sum_cost,node_cost,search=True):
    edge_indices = create_edge_indices(len(G0))  # (num_edges, 2)
    # 模型路径
    model_dir = "./Trained_models/random_cost/"
    out_dir = "./Design_result/real_network_result/random_cost/"
    model_file_names = os.listdir(model_dir)
    result_file_names =  os.listdir(out_dir)
    # 设计场景
    attack_types =['HDA','HBA','CI2','MS','GND','GNDR']
    for attack in attack_types: # 对于每一种攻击
        R_G_list = []
        if 'cost_%s_%s.pth' % (real_name, attack) in result_file_names:
            print('Finish %s %s' % (real_name, attack))
            continue
        else:
            print('Begin %s %s' % (real_name, attack))
        # 模型加载
        for trained_model_name in model_file_names:
            if attack in trained_model_name:
                model_path = model_dir + trained_model_name
            else:
                continue
            best_model = torch.load(model_path + '/best_model.pth')
            agent = PPO_cost(best_model[3], creat_file=False)
            agent.actor_net.load_state_dict(best_model[1][0])
            agent.critic_net.load_state_dict(best_model[1][1])
            agent.path = model_path
            agent.actor_net.edge_indices = edge_indices

            R_G= evaluate_policy(copy.deepcopy(G0), agent,node_cost,sum_cost,attack)
            R_G_list.append(R_G+[trained_model_name])

        if search==True:
            G_list = [x[2] for x in R_G_list]
            search_R_G = local_search_cost(G_original, G_list, attack,sum_cost, node_cost)
            R_G_list.append(search_R_G + ['search'])

        torch.save(R_G_list,out_dir+'cost_%s_%s.pth' % (real_name, attack))
        print('Finish %s %s' % (real_name, attack))



if __name__=='__main__':
    sum_cost=300
    """生成待设计的网络"""
    inpath ="../Data/real_network/"
    names = ["ISP","US Air",'Central Chilean power grid','Deltacom','Germany grid','Kdl','new-york','savannah','washington','euroroad']
    for real_name in names:
        print(real_name)
        with open(inpath+"%s_data.pkl" % real_name,"rb") as f:
            Ratio_G_dict = pickle.load(f)
        G_original = Ratio_G_dict[0]
        n = len(G_original)
        np.random.seed(0)
        node_cost = list(np.random.uniform(1, 5, n))
        main(G_original,real_name,sum_cost,node_cost,search=True)