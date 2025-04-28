import argparse
import torch
from RL_Algorithm.Environment.dismantlers.dismantlers_ import dismantle
from RL_Algorithm.model.PPO import PPO
import copy
from itertools import count
from Environment.envs import env
import os
import statistics
from RL_Algorithm.utils.base_utils import time_before_2025
from RL_Algorithm.utils.rl_utils import edge_mask, action_to_edge
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_policy(args, env, agent,virtual_node_g,add_edges=100):
    graph_ids = list(range(100))
    # action_dict = env.action_dict
    R_G_list = []
    for id_ in graph_ids:
        print(id_)
        G0,_= env.reset(id_,infer=True)
        if args.graph_type in ['BA', 'ER']:
            add_edges = len(G0)+100-len(G0.edges())
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
    graph_types = ['BA','ER','BA4','ER4','SF2.1','SF2.5']
    attack_types =['HDA','HBA','CI2','MS','GND','GNDR']
    n=100
    for graph_type in graph_types: # 对于每一类网络进行设计
         # 图1结果：ER网络，图2结果BA网络
        for attack in attack_types: # 对于每一种攻击
            path_list = []
            for trained_model_name in file_names:
                gt = 'SF' if graph_type in ['SF2.1','SF2.5'] else graph_type
                if attack+'_'+gt+'_'+str(n)+'_' in trained_model_name:
                    path_list.append(model_path+trained_model_name)
                    break
            if path_list == []:
                for trained_model_name in file_names:
                    if attack + '_' + 'BA' + '_' + str(n) in trained_model_name:
                        path_list.append(model_path + trained_model_name)
                        break
            path = path_list[0]
            if time_before_2025(path):
                from RL_Algorithm.utils.rl_utils import virtual_node_g_old as virtual_node_g
            else:
                from RL_Algorithm.utils.rl_utils import virtual_node_g
            best_model = torch.load(path+'/best_model.pth')
            agent = PPO(best_model[3],creat_file=False)
            agent.actor_net.load_state_dict(best_model[1][0])
            agent.critic_net.load_state_dict(best_model[1][1])
            agent.path = path
            # 环境设置
            args.dismantling_name= attack
            args.graph_type = graph_type
            args.dismantling_number = 1 if attack not in ['GND','GNDR'] else 4
            args.num_nodes = n
            env = Env(args)
            env.path = agent.path+"/"
            # # 批量推断
            try:
                R_G_list= torch.load(out_path+"N%s/%s_%s_%s_rl_result.pth"%(n,attack,graph_type,n))
            except:
                R_G_list = evaluate_policy(args, env, agent,virtual_node_g)
                # # 保存结果
                torch.save(R_G_list,out_path+"N%s/%s_%s_%s_rl_result.pth"%(n,attack,graph_type,n))

            Robustness_score = [x[0] for x in R_G_list]
            # 计算标准差
            std_dev = statistics.stdev(Robustness_score)
            mean = sum(Robustness_score) / len(Robustness_score)
            print(attack,graph_type,n)
            print('average:',mean)
            print('std:',std_dev)

if __name__=='__main__':
    parser = argparse.ArgumentParser("")
    args = parser.parse_args()
    env = env
    main(args, env)