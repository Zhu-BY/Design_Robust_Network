'''
修改计算方式，降低内存和显存需求:拉平流式 20251020
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers.normalize import RunningMeanStd
import numpy as np
from torch.distributions import Categorical

# from Environments.envs import Env1_tree_add_edge
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

        # 建立节点特征到边特征的映射矩阵
        self.node_to_edge_matrix1 = None
        self.node_to_edge_matrix2 =  None

        self.edge_indices = None
        

        self.num_nodes = -1
        # self.M = -1 # 侯选边数量
        self.device = device

        self.connected_keys = torch.empty(0, dtype=torch.long, device=device)

    def reset_connected_edges(self, G):
        self.num_nodes = len(G)
        N = self.num_nodes
        if G.number_of_edges() > 0:
            edges = torch.tensor(list(G.edges()), dtype=torch.long, device=self.device)
            u, v = torch.min(edges, dim=1).values, torch.max(edges, dim=1).values
            self.connected_keys = u * N + v
            self.connected_keys, _ = torch.sort(self.connected_keys)
        else:
            self.connected_keys = torch.empty(0, dtype=torch.long, device=self.device)

    def update_connected_edges(self, edge):
        u, v = edge
        N = self.num_nodes
        key = torch.tensor([min(u, v) * N + max(u, v)], dtype=torch.long, device=self.device)
        self.connected_keys = torch.cat([self.connected_keys, key])
        self.connected_keys, _ = torch.sort(self.connected_keys)
        
    @torch.no_grad()
    def find_best_edge_flat_stream(self, emb_node, graph_embeding,
                                   chunk: int = 10_000_000, subchunk: int = 10_000_000):
        chunk = self.num_nodes
        subchunk = self.num_nodes
        device = emb_node.device
        N = emb_node.size(0)
        M = N * (N - 1) // 2
    
        best_score = -float('inf')
        best_edge = (0, 0)
    
        # 图级嵌入常量
        if getattr(self, "graph_embed", None) in ("mean", "sum", "virtual"):
            if self.graph_embed == 'mean':
                emb_graph = emb_node.mean(0, keepdim=True)
            elif self.graph_embed == 'sum':
                emb_graph = emb_node.sum(0, keepdim=True)
            else:
                emb_graph = graph_embeding.unsqueeze(0)
        else:
            emb_graph = None
    
        # connected_keys 要求有序；外部如果已保证有序，这里就不再排序了
        ck_sorted = self.connected_keys
    
        # 辅助：S(u) = u*(2N-u-1)//2
        def S(u: torch.Tensor) -> torch.Tensor:
            return (u * (2 * N - u - 1)) // 2
    
        a = 2 * N - 1  # 常量
    
        for t0 in range(0, M, chunk):
            t_end = min(t0 + chunk, M)
            t = torch.arange(t0, t_end, device=device, dtype=torch.long)  # [T]
    
            # ---- 反解 t -> (u,v)，带稳健校正 ----
            t64 = t.to(torch.float64)
            a64 = torch.tensor(a, dtype=torch.float64, device=device)
            disc = a64 * a64 - 8.0 * t64  # 保证非负
            disc = torch.clamp_min(disc, 0.0)
            u = torch.floor((a64 - torch.sqrt(disc)) / 2.0).to(torch.long)
    
            # 夹到 [0, N-2]，避免极端精度误差
            u = torch.clamp(u, 0, N - 2)
    
            Su = S(u)
            # 如果 Su > t，u 偏大，往下调 1
            mask_lo = Su > t
            if mask_lo.any():
                u[mask_lo] -= 1
                u = torch.clamp(u, 0, N - 2)
                Su = S(u)
    
            # 如果 S(u+1) <= t，u 偏小，往上调 1
            Su1 = S(torch.clamp(u + 1, 0, N - 1))
            mask_hi = Su1 <= t
            if mask_hi.any():
                u[mask_hi] += 1
                u = torch.clamp(u, 0, N - 2)
                Su = S(u)
    
            off = t - Su
            v = u + 1 + off  # 必须满足 u < v
            # 最终保证边界
            v = torch.clamp(v, 0, N - 1)
    
            # 额外健壮性检查（开发期可以保留，稳定后可去掉）
            # 如果你想立即定位越界，可开启下两行：
            # assert int(u.min()) >= 0 and int(u.max()) <= N-2
            # assert int(v.min()) >= 1 and int(v.max()) <= N-1
    
            # ---- 过滤已存在边：安全版 searchsorted ----
            cand_keys = u * N + v  # [T]
            if ck_sorted.numel():
                idx = torch.searchsorted(ck_sorted, cand_keys)
                # 只在 idx 有效的位置做索引，避免 OOB
                mask_valid = idx < ck_sorted.numel()
                hit = torch.zeros_like(mask_valid, dtype=torch.bool, device=device)
                if mask_valid.any():
                    hit_valid = (ck_sorted[idx[mask_valid]] == cand_keys[mask_valid])
                    hit[mask_valid] = hit_valid
                keep = ~hit
                if not keep.any():
                    # 本批全是已有边，跳过
                    del t, u, v, cand_keys, Su, Su1, off, idx, mask_valid, hit, keep
                    torch.cuda.empty_cache()
                    continue
                u = u[keep]; v = v[keep]
    
            # ---- 真正前向：子分块 ----
            for i in range(0, u.numel(), subchunk):
                ui = u[i:i + subchunk]
                vi = v[i:i + subchunk]
                if ui.numel() == 0:
                    continue
    
                # 再次防呆（极端情况下可以保留）
                # assert int(ui.min()) >= 0 and int(vi.min()) >= 0
                # assert int(ui.max()) < N and int(vi.max()) < N
    
                v_emb = emb_node[ui]
                u_emb = emb_node[vi]
    
                if self.edge_embed == 'minus_add':
                    emb_edge = torch.cat((v_emb - u_emb, v_emb + u_emb), dim=1)
                elif self.edge_embed == 'concat':
                    emb_edge = torch.cat((v_emb, u_emb), dim=1)
                else:
                    raise ValueError(f"Unknown edge_embed: {self.edge_embed}")
    
                if emb_graph is not None:
                    emb_edge = torch.cat(
                        (emb_edge, emb_graph.expand(emb_edge.size(0), -1)),
                        dim=1
                    )
    
                h3 = self.re_activation(self.fc2(emb_edge))
                scores = self.out_action(h3).squeeze(-1)
    
                m, idx_local = torch.max(scores, dim=0)
                if m.item() > best_score:
                    ui_best = ui[idx_local].item()
                    vi_best = vi[idx_local].item()
                    # ui_best < vi_best（由构造保证），不会自环
                    best_score = m.item()
                    best_edge = (ui_best, vi_best)
    
                del v_emb, u_emb, emb_edge, h3, scores
                torch.cuda.empty_cache()
    
            del t, u, v, cand_keys, Su, Su1, off
            torch.cuda.empty_cache()
    
        return best_edge, best_score


    @torch.no_grad()
    def forward(self, g, node_features):
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

        best_edge, best_score = self.find_best_edge_flat_stream(emb_node, graph_embeding)
        
        return best_edge, best_score

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
            self.re_activation = nn.LeakyReLU()
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
