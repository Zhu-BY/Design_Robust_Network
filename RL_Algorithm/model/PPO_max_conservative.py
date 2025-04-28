'''
generate graph from a tree,
GCN用每两个节点的表示的拼接表示待连接的边特征，来预测边的策略选择，
action输出一个数，对应于两个节点

ppo_max:
1 advantage normalization
2 reward scailing
'''
import sys
sys.path.append('/root/autodl-tmp/tmp/Optimal_Graph_Generation')
from collections import namedtuple
from itertools import count
import os, time
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
# from Environments.envs import Env1_tree_add_edge
from RL_Algorithms.rl_utils import graph_batch,graph_batch_with_length
from Environments.utils import draw_G
import dgl
from dgl.nn import GraphConv
# from RL_Algorithms.Virtual_Node_Embedding.gnn_net import Actor_no_weight,Critic,RewardScaling
from RL_Algorithms.Virtual_Node_Embedding.gnn_net_optimal import Actor_no_weight,Critic,RewardScaling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO():
    def __init__(self, args,creat_file = True):
        super(PPO, self).__init__()
        # self.gamma = 0.99  # discount factor
        # self.clip_param = 0.2  # clip parameter
        # self.max_grad_norm = 0.5  # grad norm
        # self.ppo_update_time = 10  # ppo 更新次数
        # self.minibatch_size = m - n
        # self.entropy_coef = 0.01
        # self.max_train_steps = 1000
        # self.lr_a = 0.001
        # self.lr_c = 0.001
        self.gamma = args.gamma  # discount factor
        self.lamda = args.lamda # GAE parameter
        self.clip_param = args.clip_param  # clip parameter
        self.max_grad_norm = 0.5  # grad norm
        self.ppo_update_time = args.update_time # ppo 更新次数
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.entropy_coef = args.entropy_coef
        self.max_train_steps = args.max_episode_steps*args.update_time*(args.batch_size/self.mini_batch_size) # 用来更新学习率
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.value_first=args.value_first # 先訓練valuenetwork的次數
        self.actor_net = Actor_no_weight(args).to(device)
        self.critic_net = Critic(args).to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0  # 参数更新次数
        self.nowtime = str(time.localtime()[0:5])
        # CQL约束系数
        self.alpha = 0.1

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_a,eps=1e-5)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_c,eps=1e-5)
        if creat_file:
            try:
                if args.general==True:
                    if not os.path.exists('/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/%s_%s_%s_%s/param' % (args.dismantling_name,args.graph_type,args.num_nodes,self.nowtime)):
                        os.makedirs('/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/%s_%s_%s_%s/param/net_param' % (args.dismantling_name,args.graph_type,args.num_nodes,self.nowtime))
                        os.makedirs('/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/%s_%s_%s_%s/param/img' % (args.dismantling_name,args.graph_type,args.num_nodes,self.nowtime))
                    self.path ="/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/%s_%s_%s_%s"% (args.dismantling_name,args.graph_type,args.num_nodes,self.nowtime)
                if args.general=='Pre Train':
                    if not os.path.exists('/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/Pre_%s_%s_%s_%s/param' % (args.dismantling_name,args.graph_type,args.num_nodes,self.nowtime)):
                        os.makedirs('/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/Pre_%s_%s_%s_%s/param/net_param' % (args.dismantling_name,args.graph_type,args.num_nodes,self.nowtime))
                        os.makedirs('/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/Pre_%s_%s_%s_%s/param/img' % (args.dismantling_name,args.graph_type,args.num_nodes,self.nowtime))
                    self.path ="/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/Pre_%s_%s_%s_%s"% (args.dismantling_name,args.graph_type,args.num_nodes,self.nowtime)
            except:
                if not os.path.exists('/root/autodl-tmptmp//Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/%s_%s/param'%(args.num_nodes,self.nowtime)):
                    os.makedirs('/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/%s_%s/param/net_param'%(args.num_nodes,self.nowtime))
                    os.makedirs('/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/%s_%s/param/img'%(args.num_nodes,self.nowtime))
                self.path = "/root/autodl-tmp/tmp/Optimal_Graph_Generation/RL_Algorithms/Virtual_Node_Embedding/%s_%s"% (args.num_nodes,self.nowtime)
            # self.writer = SummaryWriter(self.path+'/exp')

    def select_action(self, state,edge_mask_matrix,evaluation = False,returnall=False):
        # state = torch.from_numpy(state.to(device),state.ndata['feat'].to(device)).float().unsqueeze(0)
        device = self.actor_net.fc1.weight.device
        with torch.no_grad():
            action_prob = self.actor_net(state.to(device),state.ndata['feat'].to(device),edge_mask_matrix)
        if evaluation==True:
        # if True:
            if returnall==False:
                action = torch.max(action_prob, 1)[1]
            else:
                return action_prob
        else:
            c = Categorical(action_prob)
            action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        # state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state.to(device),state.ndata['feat'].to(device))
        return value.item()

    def save_param(self):
        # torch.save(self.actor_net.state_dict(), self.path+'/param/net_param/ppo_actor_net'+ str(time.time())[:10] +'.pkl')
        # torch.save(self.critic_net.state_dict(), self.path+'/param/net_param/ppo_critic_net'+ str(time.time())[:10] +'.pkl')
        return 0

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def multi_store_transition(self, bf_transition):
        self.buffer=self.buffer+bf_transition
        self.counter += 1

    def update(self, i_ep):  # 多个trajectory训练
        state = [t.state[0] for t in self.buffer]
        edge_mask_matrixes = [t.state[1] for t in self.buffer]
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)
        reward = [t.reward for t in self.buffer]
        done = [t.done for t in self.buffer]
        next_state = [t.next_state[0] for t in self.buffer]
        next_edge_mask_matrixes = [t.next_state[1] for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(device)

        # batch_state, batch_state_features, batch_edge_mask_matrixes= graph_batch(state, edge_mask_matrixes)
        # next_batch_state, next_batch_state_features, next_batch_edge_mask_matrixes = graph_batch(next_state, next_edge_mask_matrixes)
        r = torch.tensor(reward, dtype=torch.float).view(-1, 1).cpu()
        dw = torch.tensor(done, dtype=torch.float).view(-1, 1).cpu()
# 
        # adv=[]
        # gae=0
        # with torch.no_grad():  # adv and v_target have no gradient
        #     vs = self.critic_net(batch_state.to(device), batch_state_features.to(device)).cpu()
        #     vs_ = self.critic_net(next_batch_state.to(device), next_batch_state_features.to(device)).cpu()
        #     deltas = r + self.gamma * vs_ * (1.0 - dw) - vs
        #     for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(dw.flatten().numpy())):
        #         if d==1:gae=0  # 每一个trajectory重新计算
        #         gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
        #         adv.insert(0, gae)
        #     adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
        #     v_target = adv + vs
        #     # if self.use_adv_norm:  # Trick 1:advantage normalization
        #     norm_adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        """分批次计算Vs,Vs_ """
        gae=0
        adv=[]
        with torch.no_grad():  # adv and v_target have no gradient
            mini_batch = 500 # 500
            num_batch = len(self.buffer) // mini_batch
            values = []
            values_ = []
            for i in range(num_batch):
                i0 = i * mini_batch
                i1 = (i + 1) * mini_batch
                # Gt_index = v_target[i0:i1].view(-1, 1).to(device)
                batch_state, batch_state_features, batch_edge_mask_matrixes= graph_batch(state[i0:i1],edge_mask_matrixes[i0:i1])
                next_batch_state, next_batch_state_features, next_batch_edge_mask_matrixes = graph_batch(next_state[i0:i1], next_edge_mask_matrixes[i0:i1])
                mini_vs = self.critic_net(batch_state.to(device), batch_state_features.to(device)).cpu()
                mini_vs_ = self.critic_net(next_batch_state.to(device), next_batch_state_features.to(device)).cpu()
                values.append(mini_vs)
                values_.append(mini_vs_)
                batch_state.to('cpu'), batch_state_features.to('cpu'), batch_edge_mask_matrixes.to('cpu'),next_batch_state.to('cpu'), next_batch_state_features.to('cpu'), next_batch_edge_mask_matrixes.to('cpu')
            vs = torch.cat(values, dim=0)
            vs_ = torch.cat(values_, dim=0)

            deltas = r + self.gamma * vs_ * (1.0 - dw) - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(dw.flatten().numpy())):
                if d==1:gae=0  # 每一个trajectory重新计算
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            # if self.use_adv_norm:  # Trick 1:advantage normalization
            norm_adv = ((adv - adv.mean()) / (adv.std() + 1e-5))               

        for i in range(self.ppo_update_time):
            micro_batch_size = max(1, self.mini_batch_size // 5)  #gradient accumulation 
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.mini_batch_size, False):
                self.actor_optimizer.zero_grad()
                self.critic_net_optimizer.zero_grad()
            
                # mini-batch 划分为多个 micro-batch
                micro_batches = [index[i:i+micro_batch_size] for i in range(0, len(index), micro_batch_size)]
            
                total_action_loss = 0
                total_value_loss = 0
                
                for micro_index in micro_batches:
                    Gt_index = v_target[micro_index].view(-1, 1).to(device)
                    batch_state,batch_state_features,batch_edge_mask_matrixes=graph_batch([state[ind] for ind in micro_index],[edge_mask_matrixes[ind] for ind in micro_index])
                    V = self.critic_net(batch_state.to(device),batch_state_features.to(device))
                    # if self.training_step>self.value_first:
                    advantage = norm_adv[micro_index].view(-1,1).to(device)
                    action_probs = self.actor_net(batch_state.to(device), batch_state_features.to(device),batch_edge_mask_matrixes)  # new policy    
                    # policy entropy
                    dist_now = Categorical(action_probs)
                    dist_entropy = dist_now.entropy().view(-1, 1)
    
                    action_prob = action_probs.gather(1, action[micro_index])  # new policy
                    ratio = (action_prob / old_action_log_prob[micro_index])
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
    
                    # update actor network
                    action_loss = -torch.min(surr1, surr2)- self.entropy_coef * dist_entropy  # MAX->MIN desent
                    # self.writer.add_scalar('loss/action_loss', action_loss.mean(), global_step=self.training_step)
                    # self.actor_optimizer.zero_grad()
                    action_loss.mean().backward()
                    # nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                    # self.actor_optimizer.step()
                    
                    action_probs.to('cpu')
                    # del action_probs
                    # torch.cuda.empty_cache()
    
                    # update critic network
                    # 1 标准的MSE损失
                    value_loss = F.mse_loss(Gt_index, V)    
                    # 2 增加对value network的l2正则化(所有权重的平方和)
                    l2_reg = 0
                    for param in self.critic_net.parameters():
                        l2_reg += torch.sum(param ** 2)
                    lambda_reg=0.01
                    value_loss += lambda_reg * l2_reg # 将L2正则化项加到损失中    
                    # 3 计算 CQL 正则项 (限制 V(s) 不要⾼估)
                    next_batch_state, next_batch_state_features, next_batch_edge_mask_matrixes = graph_batch([next_state[ind] for ind in micro_index], [next_edge_mask_matrixes[ind] for ind in micro_index])
                    batch_r = r[micro_index].to(device)
                    batch_dw = dw[micro_index].to(device)
                    V_ = self.critic_net(next_batch_state.to(device), next_batch_state_features.to(device)).detach()
                    q_values = batch_r + self.gamma * V_ * (1.0 - batch_dw)
                    cql_loss = self.alpha*( q_values - V).mean()    
                    value_loss+=cql_loss    
                    # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                    # self.critic_net_optimizer.zero_grad()
                    value_loss.backward()
                    # nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                    # self.critic_net_optimizer.step()

                    # 管理内存
                    batch_state.to('cpu'), batch_state_features.to('cpu'), batch_edge_mask_matrixes.to('cpu'),next_batch_state.to('cpu'), next_batch_state_features.to('cpu'), next_batch_edge_mask_matrixes.to('cpu')
                    del action_probs, batch_state, batch_state_features, batch_edge_mask_matrixes,next_batch_state, next_batch_state_features, next_batch_edge_mask_matrixes
                    torch.cuda.empty_cache()

                
                # del next_batch_state, next_batch_state_features, next_batch_edge_mask_matrixes,V_,V
                # torch.cuda.empty_cache()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                torch.cuda.empty_cache()
                self.training_step += 1
                
        with torch.no_grad():
            total_ppo_loss,total_value_mse = 0,0
            mini_batch = 100 # 500
            num_batch = len(self.buffer) // mini_batch
            for i in range(num_batch):
                i0 = i * mini_batch
                i1 = (i + 1) * mini_batch
                Gt_index = v_target[i0:i1].view(-1, 1).cpu()
                batch_state, batch_state_features, batch_edge_mask_matrixes= graph_batch(state[i0:i1],edge_mask_matrixes[i0:i1])
                V = self.critic_net(batch_state.to(device), batch_state_features.to(device)).cpu()
                advantage = norm_adv[i0:i1].view(-1, 1).to(device)
                # epoch iteration, PPO core!!!
                action_probs = self.actor_net(batch_state.to(device), batch_state_features.to(device),batch_edge_mask_matrixes.to(device))  # new policy
                # policy entropy
                try:
                    dist_now = Categorical(action_probs)
                except:
                    print('1')
                dist_entropy = dist_now.entropy().view(-1, 1)
                # a_logprob_now = dist_now.log_prob(action[:].squeeze()).view(-1, 1)
                action_prob = action_probs.gather(1, action[i0:i1])  # new policy
                ratio = (action_prob / old_action_log_prob[i0:i1])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                # actor network loss
                action_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss.mean(), global_step=self.training_step)
                # critic network loss
                value_loss = F.mse_loss(Gt_index, V)
                
                total_ppo_loss+=action_loss.mean()
                total_value_mse+=value_loss
                batch_state.to('cpu'), batch_state_features.to('cpu'), batch_edge_mask_matrixes.to('cpu')
            # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
            total_ppo_loss /= num_batch
            total_value_mse /= num_batch
            # 绘制V和Gt的散点图
            import matplotlib.pyplot as plt
            x = Gt_index.cpu().detach().numpy()
            y = V.cpu().detach().numpy()
            plt.figure()
            plt.scatter(x, y, color='blue', alpha=0.6)
            plt.xlabel("Gt values")
            plt.ylabel("V values")
            plt.grid(True)
            plt.savefig(self.path+'/value-Gt.jpg',dpi=600)
            plt.close()
            # if value_loss>1:
            #     print(1)
            # if self.training_step % 100 == 0:  # 1000
            #     print('I_ep {} ，train {} times , action loss {:.4f} , valu


        del self.buffer[:]  # clear experience
        self.lr_decay(self.training_step)
        # return self.training_step,action_loss.mean(),value_loss
        return self.training_step,total_ppo_loss,total_value_mse

    def lr_decay(self, train_steps):
        lr_a_now = self.lr_a * (1 - train_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - train_steps / self.max_train_steps)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_net_optimizer.param_groups:
            p['lr'] = lr_c_now


