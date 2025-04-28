import torch
from RL_Algorithm.Environment.dismantlers.dismantlers_ import dismantle
from RL_Algorithm.model.PPO import PPO
from RL_Algorithm.model.local_search import local_search
import random as rd
import copy
from itertools import count
import pickle
import os
from RL_Algorithm.utils.rl_utils import edge_mask, action_to_edge, create_edge_indices
from RL_Algorithm.utils.base_utils import load_csv_net,time_before_2025
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def design_network(G0,agent,termination,attack,virtual_node_g):
    target_size = 1 if attack not in ['GND', 'GNDR'] else 4
    G = copy.deepcopy(G0)
    # action_list = []
    for t in count():
        state = virtual_node_g(G)
        edge_mask_matrix = edge_mask(G)
        action, action_prob = agent.select_action(state,edge_mask_matrix,evaluation=True)
        # act = action_dict[action]
        act = action_to_edge(action, len(G0))
        # action_list.append(act)
        G.add_edge(act[0],act[1])
        if len(G.edges()) >= termination:
            episode_reward,curve,_ = dismantle(attack,G,target_size,agent.path+"/")
            break
    R_G=[episode_reward,curve,copy.deepcopy(G),copy.deepcopy(G0)]
    return R_G

def main(G_dict,real_name,search=True):
    # 模型路径
    model_dir = "./Trained_models/uniform_cost/"
    out_dir = "./Design_result/real_network_result/uniform_cost/%s/"%real_name
    model_file_names = os.listdir(model_dir)
    result_file_names =  os.listdir(out_dir)
    # 设计场景
    # attack_types =['HDA','HBA','CI2','MS','GND','GNDR']
    # ratios = [0.1,0.2,0.3,0.4,0.5]
    attack_types =['MS']
    ratios = [0.5]
    # 数据加载
    G_original = G_dict[0]
    original_edges = len(G_original.edges())
    edge_indices = create_edge_indices(len(G_original))  # (num_edges, 2)
    for ratio in ratios:
        G0 = G_dict[ratio]
        for attack in attack_types: # 对于每一种攻击
            R_G_list = []
            if '%s_%s_%s.pth' % (real_name, ratio, attack) in result_file_names:
                print('Finish %s %s %s' % (real_name, ratio, attack))
                continue
            else:
                print('Begin %s %s %s' % (real_name, ratio, attack))
            for trained_model_name in model_file_names:
                if attack in trained_model_name:
                # if trained_model_name=="HDA_BA_100_(2024, 6, 13, 18, 45)":
                    model_path = model_dir + trained_model_name
                    # print(model_path)
                    if time_before_2025(model_path):
                        from RL_Algorithm.utils.rl_utils import virtual_node_g_old as virtual_node_g
                    else:from RL_Algorithm.utils.rl_utils import virtual_node_g
                else: continue
                best_model = torch.load(model_path+ '/best_model.pth')
                agent = PPO(best_model[3],creat_file=False)
                agent.actor_net.load_state_dict(best_model[1][0])
                agent.critic_net.load_state_dict(best_model[1][1])
                agent.path = model_path
                agent.actor_net.edge_indices = edge_indices
                if ratio==0:
                    R_G= design_network(copy.deepcopy(G0), agent,original_edges+100,attack,virtual_node_g=virtual_node_g)
                else:
                    R_G = design_network(copy.deepcopy(G0), agent, original_edges,attack,virtual_node_g=virtual_node_g)
                R_G_list.append(R_G)

            if search == True:
                G_list = [x[2] for x in R_G_list]
                searched_R_G = local_search(G0, G_list, attack)
                R_G_list.append(searched_R_G+['search'])

            torch.save(R_G_list,out_dir+'/%s_%s_%s.pth'%(real_name,ratio,attack))
            print('Robustness:',max([x[0] for x in R_G_list]))
            print('Finish %s %s %s' % (real_name, ratio, attack))


if __name__=='__main__':
    path = "../Data/real_network/"
    # names = ["ISP","US Air",'Central Chilean power grid','Deltacom','Germany grid','Kdl','new-york','savannah','washington','euroroad']
    names = ["US Air"]
    for real_name in names:
        try:
            # 读取 Pickle 文件
            with open(path + "%s_data.pkl"%real_name, "rb") as f:
                ratio_G_dict = pickle.load(f)
        except:
            # 生成随机删边的网络并保存
            ratio_G_dict = {}
            Graph, pos = load_csv_net(real_name)  # 真实网络图
            ratio_G_dict[0] = copy.deepcopy(Graph)
            print(real_name,len(Graph),len(Graph.edges()))
            if len(Graph)>1500:
                continue
            seed=0
            ratios = [0.1, 0.2, 0.3, 0.4, 0.5,0]
            G = copy.deepcopy(Graph)
            for i in range(5):
                # 随机删除ratio比例的边
                edge_list = list(G.edges())
                n_remove = int(len(edge_list) * 0.1)  # 删边数=添加的边数
                rd.seed(seed)
                remove_list = rd.sample(edge_list, n_remove)
                G.remove_edges_from(remove_list)
                # 删边网络属性
                G_ = copy.deepcopy(G)
                ratio_G_dict[ratios[i]] = G_
            with open(path+"%s_data.pkl"%real_name, "wb") as f:
                pickle.dump(ratio_G_dict, f)
        main(ratio_G_dict,real_name,search=False)