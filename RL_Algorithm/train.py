import sys
sys.path.append('')
import argparse
from RL_Algorithm.Environment.dismantlers.dismantlers_ import dismantle
from RL_Algorithm.model.PPO import PPO,RewardScaling
from RL_Algorithm.model.pretrain_policy_net import pretrain_actor_net
from collections import namedtuple
from itertools import count
import torch
import random as rd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import copy
from Environment.envs import env
from RL_Algorithm.utils.rl_utils import saveargs,edge_mask, virtual_node_g, action_to_edge
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_policy(args, env, agent,add_edges=100):
    train_graph_ids =args.train_graph_ids
    test_graph_ids =args.test_graph_ids
    graph_ids =train_graph_ids+test_graph_ids
    # action_dict = env.action_dict
    evaluate_reward = []
    for _ in graph_ids:
        episode_reward = 0
        G0,pos = env.reset(_)
        if args.graph_type in ['BA', 'ER']:
            add_edges = len(G0)+100-len(G0.edges())
        G = copy.deepcopy(G0)
        for t in count():
            state = virtual_node_g(G)
            edge_mask_matrix = edge_mask(G)
            action, action_prob = agent.select_action(state,edge_mask_matrix,evaluation=True)
            act = action_to_edge(action,args.num_nodes)
            G.add_edge(act[0],act[1])
            if len(G.edges()) >= len(G0.edges)+add_edges:
                episode_reward,_,_ = dismantle(args.dismantling_name,G,args.dismantling_number,agent.path+"/")
                break
        evaluate_reward.append(episode_reward)
    train_evaluate_reward = sum(evaluate_reward[0:len(train_graph_ids)]) / len(train_graph_ids)
    test_evaluate_reward = sum(evaluate_reward[len(train_graph_ids):]) / len(test_graph_ids)
    return train_evaluate_reward, test_evaluate_reward

def main(args,Env):
    best_model = [0,None,0,args] # epoch,model,value
    agent = PPO(args)
    if args.pretrain==True:
        data_size=100
        pretrain_path = "./model/pretrain/"
        try:
            try:
                pre_train_dict = torch.load(pretrain_path+'Pretrain_policy_%s_%s_%s.pth'%(data_size,args.graph_type,args.num_nodes))[0]
            except:
                pre_train_dict = torch.load(pretrain_path+'Pretrain_policy_%s_%s_%s.pth'%(data_size,args.graph_type,args.num_nodes))
        except:
            pre_train_dict = pretrain_actor_net(args,100,path =pretrain_path )
        agent.actor_net.load_state_dict(pre_train_dict)
    # 加载环境
    env = Env(args,agent.path+"/")
    env_evaluate = Env(args,agent.path+"/")
    # PPo Parameters
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state','done'])
    TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
    saveargs(args,agent.path)
    reward_scailing = RewardScaling(1, args.gamma)
    training_records = []
    evaluate_num = 0
    train_evaluate_rewards = []
    test_evaluate_rewards = []
    action_loss_list = []
    value_loss_list = []
    for i_epoch in range(args.max_episode_steps):
        seed = rd.randint(args.train_seed[0], args.train_seed[1])
        G,pos = env.reset(seed)
        score = 0
        reward_scailing.reset()
        for t in count():
            state = virtual_node_g(G)
            edge_mask_matrix = edge_mask(G)
            action, action_prob = agent.select_action(state,edge_mask_matrix)
            act = action_to_edge(action,args.num_nodes)
            reward, next_G,done = env.step(act)
            score += reward
            reward = reward_scailing(reward)
            next_state = virtual_node_g(next_G)
            next_edge_mask_matrix = edge_mask(next_G)
            trans = Transition([state.cpu(),edge_mask_matrix.cpu()], action, action_prob, reward, [next_state.cpu(),next_edge_mask_matrix.cpu()],done)
            agent.store_transition(trans)
            G = next_G

            if len(agent.buffer) >= args.batch_size:
                training_step, action_loss, value_loss = agent.update(i_epoch)
                action_loss_list.append(action_loss.cpu().detach().numpy())
                value_loss_list.append(value_loss.cpu().detach().numpy())
                print("ep:{} \t train {} times \t action loss {:.4f} \t value loss {:.4f}".format(i_epoch,agent.training_step,action_loss,value_loss))
                # Evaluate the policy every 'evaluate_freq' steps
                if i_epoch % args.evaluate_freq == 0 and i_epoch>0:
                    evaluate_num += 1
                    train_evaluate_reward,test_evaluate_reward = evaluate_policy(args, env_evaluate, agent)
                    train_evaluate_rewards.append(train_evaluate_reward);test_evaluate_rewards.append(test_evaluate_reward)
                    print("ep:{}  eva_num:{}\ttrain_r:{:.2f} \ttest_r:{:.2f} \t train {} times \t action loss {:.4f} \t value loss {:.4f}".format(i_epoch,evaluate_num, train_evaluate_reward,test_evaluate_reward,agent.training_step,action_loss,value_loss))
                    if train_evaluate_reward+test_evaluate_reward>best_model[2]:
                        best_model[2]=train_evaluate_reward+test_evaluate_reward
                        best_model[1]= [copy.deepcopy(agent.actor_net.state_dict()),copy.deepcopy(agent.critic_net.state_dict())]
                        best_model[0] = i_epoch
                    if i_epoch % args.show_evaluate_freq== 0:
                        plt.plot([args.evaluate_freq*i for i in range(1,evaluate_num+1)], train_evaluate_rewards,label='train')
                        plt.plot([args.evaluate_freq * i for i in range(1, evaluate_num + 1)], test_evaluate_rewards, label='test')
                        plt.title('PPO reward with epoch'); plt.xlabel('epoch'); plt.ylabel('reward');plt.legend()
                        plt.savefig(agent.path+"/param/img/reward_ep_%s.png"%(i_epoch))
                        plt.close()

                        plt.plot( value_loss_list,label='value_loss')
                        plt.title('Value loss');plt.xlabel('epoch'); plt.ylabel('loss');plt.legend()
                        plt.savefig(agent.path+"/param/img/value_loss_%s.png"%(i_epoch))
                        plt.close()

                        plt.plot( action_loss_list,label='action_loss')
                        plt.title('Action loss');plt.xlabel('epoch'); plt.ylabel('loss');plt.legend()
                        plt.savefig(agent.path+"/param/img/action_loss_%s.png"%(i_epoch))
                        plt.close()
                agent.save_param()
                torch.save(best_model,agent.path+'/best_model.pth')
            if done:
                break

        running_reward = score
        training_records.append(TrainingRecord(i_epoch, running_reward))
    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Episode reward')
    plt.savefig(agent.path+"/param/img/ppo_all_episode.png")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO of graph generation")
    # 环境参数
    parser.add_argument("--graph_type", type=str, default='BA', help="ER,BA,zero,Tree,WS0.3,WS0.1,WS0.5")
    parser.add_argument("--dismantling_name", type=str, default='HDA', help="HBA,HDA,CI2,MS,GND,GNDR")
    parser.add_argument("--dismantling_number", type=int, default=int(1), help="瓦解目标size")
    # parser.add_argument("--num_edges", type=int, default=None, help="新增连边数：100/2倍节点数-初始边数")
    parser.add_argument("--train_seed",  default=[100,150])
    parser.add_argument("--train_graph_ids", type=list, default=[100,101,102,103,104])
    parser.add_argument("--test_graph_ids", type=list, default=[150,151,152,153,154])
    parser.add_argument("--reward_type", type=str, default="slope", help="slope,area1,area2,area3")

    parser.add_argument("--num_nodes", type=int, default=int(100), help=" The number of nodes in the network")
    parser.add_argument("--pretrain", type=bool, default=True, help="Generality of training")
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

    parser.add_argument("--conservative", type=bool, default=True, help="Whether conservative learning for critic")
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
    env = env
    main(args, env)