'''
修改计算方式，降低内存和显存需求
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

import dgl
from dgl.nn.pytorch import GraphConv,SAGEConv,GATv2Conv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Actor(nn.Module):  # 图神经网络_no batch
    """docstring for Net"""
    def __init__(self, args):
        super(Actor, self).__init__()
        if args.gcn_activation =='relu': self.gcn_activation = nn.ReLU()
        if args.gcn_activation == 'elu': self.gcn_activation = nn.ELU()
        if args.gcn_activation == 'tanh': self.gcn_activation = nn.Tanh()
        if args.re_activation =='relu': self.re_activation = nn.ReLU()
        if args.re_activation == 'elu': self.re_activation = nn.ELU()
        if args.re_activation == 'tanh': self.re_activation = nn.Tanh()
        # if args.activation == 'leakyrelu': self.activation = nn.LeakyReLU
        self.node_norm2 = args.node_embed_norm2
        self.graph_embed = args.graph_embed
        self.edge_embed = args.edge_embed
        try:
            self.dropout = nn.Dropout(p=args.dropout)
        except:
            self.dropout = nn.Dropout(p=0)
        # 第一个全连接层
        self.fc1 = nn.Linear(args.feat_dim,args.hidden_dim1)
        # 图卷积层
        self.num_layers = args.num_layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_linear_layer = nn.ModuleList()

        if args.gnn_type == 'GATv2':
            self.gnn_type = 'GATv2'
            num_layer = self.num_layers
            num_heads = args.num_heads  # 1,2,8
            num_out_heads = 1
            heads = ([num_heads] * (num_layer - 1)) + [num_out_heads]
            self.gcn_layers.append(
                GATv2Conv(args.hidden_dim1,args.hidden_dim1, num_heads=heads[0], bias=True, activation=None,allow_zero_in_degree=True))  # 第一层
            for i in range(1, args.num_layers - 1):
                self.gcn_layers.append(
                    GATv2Conv(args.hidden_dim1 * heads[i - 1], args.hidden_dim1, num_heads=heads[i], bias=True, activation=None,allow_zero_in_degree=True))  # 中间层
            self.gcn_layers.append(
                GATv2Conv(args.hidden_dim1 * heads[-2], args.hidden_dim1, num_heads=heads[-1], bias=True, activation=None,allow_zero_in_degree=True))  # 最后一层

            for i in range(0, args.num_layers):
                self.gcn_linear_layer.append(nn.Linear(self.gcn_layers[i].fc_src.in_features, self.gcn_layers[i].fc_src.out_features))

        # 对node embedding输出的立刻全连接层
        self.fc_after_gcn = nn.Linear(args.hidden_dim1,args.hidden_dim1)

        # 第二、三个全连接层
        if self.graph_embed:
            self.fc2 = nn.Linear(args.hidden_dim1*3, args.hidden_dim2)
        else:
            self.fc2 = nn.Linear(args.hidden_dim1 * 2, args.hidden_dim2)
        self.out_action = nn.Linear(args.hidden_dim2,1)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.out_action, gain=0.01)

        # # 建立节点特征到边特征的映射矩阵
        # B,C = [],[]
        # for i in range(args.num_nodes - 1):
        #     for j in range(i + 1, args.num_nodes):
        #         b,c = [],[]
        #         for k in range(args.num_nodes):
        #             if k == i:            b.append(1)
        #             if k != i:            b.append(0)
        #             if k == j:            c.append(1)
        #             if k != j:            c.append(0)
        #         B.append(b)
        #         C.append(c)
        self.node_to_edge_matrix1 = None
        self.node_to_edge_matrix2 = None

        def create_edge_indices(n):
            edges = []
            for i in range(n - 1):
                for j in range(i + 1, n):
                    edges.append((i, j))
            edges = torch.tensor(edges, dtype=torch.long).to(device)
            return edges

        self.edge_indices = create_edge_indices(args.num_nodes)

    def forward(self, g, node_features,edge_features_mask_matrix):  # g:DGLGraph.batch  node_features:batch_size,节点数,时间序列   edge_features:batch_size,边数,值
        if len(node_features.shape) == 3:  # 批量输入
            batch_size = node_features.shape[0]
            # batch_node_features = node_features.reshape(-1, node_features.shape[-1])
            h1 = self.fc1(node_features)
            h2 = F.normalize(h1, p=2, dim=1, eps=1e-12, out=None)
            h2 = h2.reshape(-1, h2.shape[-1])
            for _ in range(self.num_layers - 1):  # 前几层：使用激活函数
                if len(self.gcn_linear_layer) == 3:
                    h2 = self.gcn_activation(self.gcn_layers[_](g, h2).flatten(1) + self.gcn_linear_layer[_](h2))  # 参考GDM对GAT的操作
                    h2 = F.normalize(h2.reshape(batch_size,-1, h2.shape[-1]), p=2, dim=1, eps=1e-12, out=None).reshape(-1, h2.shape[-1])
                    # h2 = self.dropout(h2)
                else:
                    h2 = self.gcn_activation(self.gcn_layers[_](g, h2))
                    h2 = F.normalize(h2.reshape(batch_size,-1, h2.shape[-1]), p=2, dim=1, eps=1e-12, out=None).reshape(-1, h2.shape[-1])
                    # h2 = self.dropout(h2)
            if len(self.gcn_linear_layer) == 3:  # 最后一层：不使用激活函数
                if self.gnn_type == 'GCN':
                    emb_node = self.gcn_layers[-1](g, h2).flatten(1) + self.gcn_linear_layer[-1](h2)
                    # emb_node = self.dropout(emb_node)
                else:
                    emb_node = self.gcn_layers[-1](g, h2).mean(1) + self.gcn_linear_layer[-1](h2)
                    # emb_node = self.dropout(emb_node)
            else:
                emb_node = self.gcn_layers[-1](g, h2)  # 节点的嵌入表示
                # emb_node = self.dropout(emb_node)
            batch_emb_node = emb_node.view(batch_size, node_features.shape[1], -1)
            batch_emb_node = F.normalize(batch_emb_node, p=2, dim=1, eps=1e-12, out=None)
            # 分离graph embedding和node embedding
            batch_graph_embeding = batch_emb_node[:,-1,:]
            batch_emb_node=batch_emb_node[:,0:-1,:]
            # 分离结束
            # 对node embedding输出的立刻全连接层
            batch_emb_node = self.fc_after_gcn(batch_emb_node)
            batch_emb_node = self.gcn_activation(batch_emb_node)
            # batch_emb_node = self.dropout(batch_emb_node)

            batch_edge_features_mask_matrix = edge_features_mask_matrix.reshape(batch_size,-1)  # mask invalid action
            # v_emb = torch.matmul(self.node_to_edge_matrix1, batch_emb_node)
            # u_emb = torch.matmul(self.node_to_edge_matrix2, batch_emb_node)

            v_emb = batch_emb_node[:, self.edge_indices[:, 0], :]  # (batch_size, num_edges, node_dim)
            u_emb = batch_emb_node[:, self.edge_indices[:, 1], :]

            if self.edge_embed == 'minus_add':
                batch_emb_edge = torch.concat((v_emb - u_emb, v_emb + u_emb), dim=2)
            if self.edge_embed == 'concat':
                batch_emb_edge = torch.concat((v_emb, u_emb), dim=2)
            # 添加结束##########
            if self.graph_embed:
                if self.graph_embed == 'mean':
                    batch_emb_graph = torch.mean(batch_emb_node, dim=1).unsqueeze(1).expand(batch_size,
                                                                                            batch_emb_edge.shape[1],
                                                                                            -1)
                if self.graph_embed == 'sum':
                    batch_emb_graph = torch.sum(batch_emb_node, dim=1).unsqueeze(1).expand(batch_size,
                                                                                           batch_emb_edge.shape[1],
                                                                                           -1)
                if self.graph_embed == 'virtual':
                    batch_emb_graph = batch_graph_embeding.unsqueeze(1).expand(batch_size,batch_emb_edge.shape[1], -1)
                batch_emb_edge = torch.concat((batch_emb_edge, batch_emb_graph), dim=2)
            ################## 屏蔽已有边###################
            batch_emb_edge_null = torch.zeros(batch_emb_edge.shape).to(device)
            batch_emb_edge = torch.where(
                torch.tile(batch_edge_features_mask_matrix.unsqueeze(-1), (1, 1, batch_emb_edge.shape[-1])),
                batch_emb_edge,
                batch_emb_edge_null)
            ##################屏蔽结束#####################

            # 全连接回归层
            h3 = self.fc2(batch_emb_edge)
            h3_ = self.re_activation(h3)
            h3_ = self.dropout(h3_)
            action_prob = self.out_action(h3_)
            action_prob = action_prob.reshape(-1, action_prob.shape[1])
            ####################action mask######################
            action_null = torch.full(action_prob.shape, -999).to(device)
            action_prob = torch.where(batch_edge_features_mask_matrix.to(device),
                                      action_prob,
                                      action_null)
            batch_edge_features_mask_matrix.to('cpu')
            ###################action mask#######################
            action_prob = action_prob.double()
            try:
                action_prob = F.softmax(action_prob)
            except:
                print(1)
            try:
                Categorical(action_prob)
            except:
                print(1)
            return action_prob  # 修正维度 # 动作价值函数
        else:
            h1 = self.fc1(node_features)  # 全连接层对节点特征进行嵌入
            # 节点嵌入
            h2 = F.normalize(h1,p=2,dim=0,eps=1e-12,out=None)
            for _ in range(self.num_layers - 1):  # 前几层：使用激活函数
                if len(self.gcn_linear_layer) == 3:
                    h2 = self.gcn_activation(self.gcn_layers[_](g, h2).flatten(1) + self.gcn_linear_layer[_](h2))
                    h2 = F.normalize(h2,p=2,dim=0,eps=1e-12,out=None)
                    # h2 = self.dropout(h2)
                else:
                    h2 = self.gcn_activation(self.gcn_layers[_](g, h2))
                    h2 = F.normalize(h2, p=2, dim=0, eps=1e-12, out=None)
                    # h2 = self.dropout(h2)
            if len(self.gcn_linear_layer) == 3:  # 最后一层：不使用激活函数
                if self.gnn_type == 'GCN':
                    emb_node = self.gcn_layers[-1](g, h2).flatten(1) + self.gcn_linear_layer[-1](h2)
                    # emb_node = self.dropout(emb_node)
                else:
                    emb_node = self.gcn_layers[-1](g, h2).mean(1) + self.gcn_linear_layer[-1](h2)
                    # emb_node = self.dropout(emb_node)
            else:
                emb_node = self.gcn_layers[-1](g, h2)  # 节点的嵌入表示
                # emb_node = self.dropout(emb_node)
            # 对node embedding输出的立刻全连接层
            emb_node_norm = F.normalize(emb_node, p=2, dim=0, eps=1e-12, out=None)
            # 分离graph embedding和node embedding
            graph_embeding = emb_node_norm[-1]
            emb_node_norm = emb_node_norm[0:-1]
            # 分离结束
            emb_node = self.fc_after_gcn(emb_node_norm)
            emb_node = self.gcn_activation(emb_node)
            # emb_node = self.dropout(emb_node)

            # v_emb = torch.matmul(self.node_to_edge_matrix1, emb_node)
            # u_emb = torch.matmul(self.node_to_edge_matrix2, emb_node)

            # v_emb = torch.bmm(self.node_to_edge_matrix1, batch_emb_node)  # 稀疏乘法
            # u_emb = torch.bmm(self.node_to_edge_matrix2, batch_emb_node)

            v_emb = emb_node[self.edge_indices[:, 0], :]  # (batch_size, num_edges, node_dim)
            u_emb = emb_node[self.edge_indices[:, 1], :] 

            if self.edge_embed == 'minus_add':
                emb_edge = torch.concat((v_emb - u_emb, v_emb + u_emb), dim=1)
            if self.edge_embed == 'concat':
                emb_edge = torch.concat((v_emb, u_emb), dim=1)
            del v_emb,u_emb,emb_node,emb_node_norm,h1,h2
            torch.cuda.empty_cache()
            if self.graph_embed:
                if self.graph_embed == 'mean':
                    emb_graph = torch.mean(emb_node, dim=0).unsqueeze(0).expand(emb_edge.shape[0], -1)
                if self.graph_embed == 'sum':
                    emb_graph = torch.sum(emb_node, dim=0).unsqueeze(0).expand(emb_edge.shape[0], -1)
                if self.graph_embed == 'virtual':
                    emb_graph = graph_embeding.unsqueeze(0).expand(emb_edge.shape[0], -1)
                emb_edge = torch.concat((emb_edge, emb_graph), dim=1)
            # 全连接回归层
            h3 = self.fc2(emb_edge)
            h3_ = self.re_activation(h3)
            h3_ = self.dropout(h3_)
            action_prob = self.out_action(h3_)
            ####################action mask######################
            action_prob = action_prob.T.masked_fill(~edge_features_mask_matrix.unsqueeze(0), -999)
            ###################action mask finish#######################
            action_prob = action_prob.double()
            action_prob = F.softmax(action_prob)
            return action_prob  # 修正维度 # 动作价值函数

class Critic(nn.Module):  # 图神经网络_no batch
    """docstring for Net"""
    def __init__(self, args):
        super(Critic, self).__init__()
        if args.gcn_activation =='relu': self.gcn_activation = nn.ReLU()
        if args.gcn_activation == 'elu': self.gcn_activation = nn.ELU()
        if args.gcn_activation == 'tanh': self.gcn_activation = nn.Tanh()
        try:
            if args.critic_re_activation =='relu': self.re_activation = nn.ReLU()
            if args.critic_re_activation == 'elu': self.re_activation = nn.ELU()
            if args.critic_re_activation == 'tanh': self.re_activation = nn.Tanh()
            if args.critic_re_activation == 'leakyrelu': self.re_activation = nn.LeakyReLU()
        except:
            if args.re_activation =='relu': self.re_activation = nn.ReLU()
            if args.re_activation == 'elu': self.re_activation = nn.ELU()
            if args.re_activation == 'tanh': self.re_activation = nn.Tanh()
            if args.re_activation == 'leakyrelu': self.re_activation = nn.LeakyReLU()
        self.node_norm2 = args.node_embed_norm2
        self.graph_embed = args.graph_embed
        try:
            self.dropout = nn.Dropout(p=args.dropout)
        except:
            self.dropout = nn.Dropout(p=0)
        # 第一个全连接层
        self.fc1 = nn.Linear(args.feat_dim,args.hidden_dim1)
        # 图卷积层
        self.num_layers = args.num_layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_linear_layer = nn.ModuleList()
        if args.gnn_type == 'GCN':
            self.gnn_type = 'GCN'
            for _ in range(args.num_layers):
                self.gcn_layers.append(
                    GraphConv(args.hidden_dim1, args.hidden_dim1,norm = args.norm,bias=True,weight=True,allow_zero_in_degree=True))
            for i in range(0, args.num_layers):
                self.gcn_linear_layer.append(nn.Linear(args.hidden_dim1, args.hidden_dim1))

        if args.gnn_type == 'GraphSage':
            self.gnn_type = 'GraphSage'
            for _ in range(args.num_layers):
                self.gcn_layers.append(
                    SAGEConv(args.hidden_dim1, args.hidden_dim1, args.aggregator_type,feat_drop=0,bias=True,activation=None))
                    # mean,lstm, gcn,pool

        if args.gnn_type == 'GATv2':
            self.gnn_type = 'GATv2'
            num_layer = self.num_layers
            num_heads = args.num_heads  # 1,2,8
            num_out_heads = 1
            heads = ([num_heads] * (num_layer - 1)) + [num_out_heads]
            self.gcn_layers.append(
                GATv2Conv(args.hidden_dim1,args.hidden_dim1, num_heads=heads[0], bias=True, activation=None,allow_zero_in_degree=True))  # 第一层
            for i in range(1, args.num_layers - 1):
                self.gcn_layers.append(
                    GATv2Conv(args.hidden_dim1 * heads[i - 1], args.hidden_dim1, num_heads=heads[i], bias=True, activation=None,allow_zero_in_degree=True))  # 中间层
            self.gcn_layers.append(
                GATv2Conv(args.hidden_dim1 * heads[-2], args.hidden_dim1, num_heads=heads[-1], bias=True, activation=None,allow_zero_in_degree=True))  # 最后一层

            for i in range(0, args.num_layers):
                self.gcn_linear_layer.append(nn.Linear(self.gcn_layers[i].fc_src.in_features, self.gcn_layers[i].fc_src.out_features))

        # 对node embedding输出的立刻全连接层
        self.fc_after_gcn = nn.Linear(args.hidden_dim1,args.hidden_dim1)

        # 第二、三个全连接层
        self.fc2 = nn.Linear(args.hidden_dim1, args.hidden_dim2)
        self.state_value = nn.Linear(args.hidden_dim2,1)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.state_value, gain=0.01)

    def forward(self, g, node_features):  # g:DGLGraph.batch  node_features:batch_size,节点数,时间序列   edge_features:batch_size,边数,值
        if self.node_norm2: # 每次输出对节点特征L2范数归一化
            if len(node_features.shape) == 3:  # 批量输入
                batch_size = node_features.shape[0]
                # batch_node_features = node_features.reshape(-1, node_features.shape[-1])
                h1 = self.fc1(node_features)
                h1 = F.normalize(h1, p=2, dim=1, eps=1e-12, out=None)
                h2 = h1.reshape(-1, h1.shape[-1])
                for _ in range(self.num_layers - 1):  # 前几层：使用激活函数
                    if len(self.gcn_linear_layer) == 3:
                        h2 = self.gcn_activation(self.gcn_layers[_](g, h2).flatten(1) + self.gcn_linear_layer[_](h2))
                        h2 = F.normalize(h2.reshape(batch_size,-1, h2.shape[-1]), p=2, dim=1, eps=1e-12, out=None).reshape(-1, h2.shape[-1])
                        # h2 = self.dropout(h2)
                    else:
                        h2 = self.gcn_activation(self.gcn_layers[_](g, h2))
                        h2 = F.normalize(h2.reshape(batch_size,-1, h2.shape[-1]), p=2, dim=1, eps=1e-12, out=None).reshape(-1, h2.shape[-1])
                        # h2 = self.dropout(h2)
                if len(self.gcn_linear_layer) == 3:  # 最后一层：不使用激活函数
                    if self.gnn_type == 'GCN':
                        emb_node = self.gcn_layers[-1](g, h2).flatten(1) + self.gcn_linear_layer[-1](h2)
                        # emb_node = self.dropout(emb_node)
                    else:
                        emb_node = self.gcn_layers[-1](g, h2).mean(1) + self.gcn_linear_layer[-1](h2)
                        # emb_node = self.dropout(emb_node)
                else:
                    emb_node = self.gcn_layers[-1](g, h2)  # 节点的嵌入表示
                    # emb_node = self.dropout(emb_node)
                batch_emb_node = emb_node.view(batch_size, node_features.shape[1], -1)
                batch_emb_node = F.normalize(batch_emb_node, p=2, dim=1, eps=1e-12, out=None)
                # 分离graph embedding和node embedding
                batch_graph_embeding = batch_emb_node[:,-1,:]

                if self.graph_embed == 'virtual':
                    batch_graph_emb = batch_graph_embeding
                # 全连接回归层
                h3 = self.fc2(batch_graph_emb)
                h3_ = self.re_activation(h3)
                # h3_ = F.elu(h3) # 最新代码使得value的输出有正有负Leaky ReLU leaky_relu selu
                h3_ = self.dropout(h3_)
                value = self.state_value(h3_)
                return value
            else:
                h1 = self.fc1(node_features)  # 全连接层对节点特征进行嵌入
                # 节点嵌入
                h2 = F.normalize(h1,p=2,dim=0,eps=1e-12,out=None)
                for _ in range(self.num_layers - 1):  # 前几层：使用激活函数
                    if len(self.gcn_linear_layer) == 3:
                        h2 = self.gcn_activation(self.gcn_layers[_](g, h2).flatten(1) + self.gcn_linear_layer[_](h2))
                        h2 = F.normalize(h2,p=2,dim=0,eps=1e-12,out=None)
                        # h2 = self.dropout(h2)
                    else:
                        h2 = self.gcn_activation(self.gcn_layers[_](g, h2))
                        h2 = F.normalize(h2,p=2,dim=0,eps=1e-12,out=None)
                        # h2 = self.dropout(h2)
                if len(self.gcn_linear_layer) == 3:  # 最后一层：不使用激活函数
                    if self.gnn_type == 'GCN':
                        emb_node = self.gcn_layers[-1](g, h2).flatten(1) + self.gcn_linear_layer[-1](h2)
                        # emb_node = self.dropout(emb_node)
                    else:
                        emb_node = self.gcn_layers[-1](g, h2).mean(1) + self.gcn_linear_layer[-1](h2)
                        # emb_node = self.dropout(emb_node)
                else:
                    emb_node = self.gcn_layers[-1](g, h2)  # 节点的嵌入表示
                    # emb_node = self.dropout(emb_node)

                # 对node embedding输出的立刻全连接层
                emb_node_norm = F.normalize(emb_node, p=2, dim=0, eps=1e-12, out=None)
                graph_embeding = emb_node_norm[-1]

                if self.graph_embed == 'virtual':
                    graph_emb = graph_embeding
                # 全连接回归层
                h3 = self.fc2(graph_emb)
                h3_ = self.re_activation(h3)
                # h3_ = F.elu(h3) # 最新代码使得value的输出有正有负Leaky ReLU
                h3_= self.dropout(h3_)
                value = self.state_value(h3_)
                return value


def orthogonal_init(layer, gain=1.0):  # 正交初始化
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

if __name__=="__main__":
    print(1)