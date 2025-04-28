import argparse
import torch
from RL_Algorithm.Environment.dismantlers.dismantlers_ import dismantle
from RL_Algorithm.model.PPO import PPO
import copy
from itertools import count
from Environment.envs import env
from utils.base_utils import time_before_2025
import os
import re
import statistics
from RL_Algorithm.utils.rl_utils import edge_mask, action_to_edge,create_edge_indices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_policy(args, env, agent,virtual_node_g,add_edges=100,):
    graph_ids = list(range(100))
    # action_dict = env.action_dict
    R_G_list = []
    for id_ in graph_ids:
        print(id_)
        G0,_ = env.reset(id_,infer=True)
        if args.n>=1000:
            add_edges = int(args.n/10)
        G = copy.deepcopy(G0)
        G_list = [G0]
        for t in count():
            state = virtual_node_g(G)
            edge_mask_matrix = edge_mask(G)
            action, action_prob = agent.select_action(state,edge_mask_matrix,evaluation=True)
            act = action_to_edge(action, args.num_nodes)
            G.add_edge(act[0],act[1])
            G_list.append(copy.deepcopy(G))
            if len(G.edges()) >= len(G0.edges)+add_edges:
                episode_reward,_,_ = dismantle(env.dismantling_name,G,env.dis_p_n,agent.path+"/")
                break
        R_G_list.append([episode_reward,copy.deepcopy(G),G_list])
    return R_G_list

def main(args,Env):
    model_path = "./Trained_models/uniform_cost/"
    out_path = "./Design_result/synthetic_result/"
    file_names = os.listdir(model_path)
    graph_types = ['BA','ER']
    attack_types =['HDA','HBA','CI2','MS','GND','GNDR']
    ns=[600,1000,5000]
    for n in ns:
        edge_indices = create_edge_indices(n)  # (num_edges, 2)
        for graph_type in graph_types: # 对于每一类网络进行设计
            for attack in attack_types: # 对于每一种攻击
                try:
                    rl_result = torch.load(out_path + "N%s/%s_%s_%s_rl_result.pth" % (n, attack, graph_type, n))
                except:
                    path_list = []
                    for trained_model_name in file_names:
                        if attack+'_' + graph_type + '_' + '1000' in trained_model_name and n==5000:
                            path_list=[model_path + trained_model_name]
                            break
                        elif attack in ['HDA','HBA','CI2']:
                            if 'HDA_' + graph_type + '_' + '100_' in trained_model_name or 'HBA_' + graph_type + '_' + '100_'in trained_model_name or 'CI2_' + graph_type + '_' + '100_' in trained_model_name :
                                path_list.append(model_path + trained_model_name)
                        elif attack+'_' + graph_type + '_' + '100_' in trained_model_name:
                            path_list.append(model_path + trained_model_name)
                    print(path_list)
                    rl_result = None
                    best_R = 0
                    for path in path_list:
                        if time_before_2025(path):
                            from RL_Algorithm.utils.rl_utils import virtual_node_g_old as virtual_node_g
                        else:from RL_Algorithm.utils.rl_utils import virtual_node_g
                        best_model = torch.load(path+'/best_model.pth')
                        agent = PPO(best_model[3],creat_file=False)
                        agent.actor_net.load_state_dict(best_model[1][0])
                        agent.critic_net.load_state_dict(best_model[1][1])
                        agent.path = path
                        agent.actor_net.edge_indices = edge_indices
                        # 环境设置
                        args.dismantling_name= attack
                        args.graph_type = graph_type
                        args.dismantling_number = 1 if attack not in ['GND','GNDR'] else 4
                        args.num_nodes = n
                        env = Env(args)
                        env.path = agent.path+"/"
                        # # 批量推断
                        R_G_list = evaluate_policy(args, env, agent,virtual_node_g)
                        sumR = sum([x[0] for x in R_G_list])
                        if sumR>best_R:
                            best_R=sumR
                            rl_result=R_G_list
                        # # 保存结果
                    torch.save(rl_result,out_path+"N%s/%s_%s_rl_result.pth"%(n,attack,graph_type,n))

                Robustness_score = [x[0] for x in rl_result]
                # 计算标准差
                std_dev = statistics.stdev(Robustness_score)
                mean = sum(Robustness_score) / len(Robustness_score)
                print(attack, graph_type, n)
                print('average:', mean)
                print('std:', std_dev)

if __name__=='__main__':
    parser = argparse.ArgumentParser("")
    args = parser.parse_args()
    env = env
    main(args, env)