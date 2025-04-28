from RL_Algorithm.Environment.dismantlers.dismantlers_ import dismantle
from RL_Algorithm.utils.base_utils import generate_network,generate_network_cost
import copy

class env(object):
    def __init__(self, args,path=0):
        self.n = args.num_nodes # 网络规模
        self.dismantling_name = args.dismantling_name # 瓦解方法
        self.graph_type = args.graph_type # 生成的网络类型
        try:
            self.penalty=args.env_penalty # 是否有添加边失败的惩罚
            self.reward_type = args.reward_type # reward的类型
        except:
            self.penalty=0 # 是否有添加边失败的惩罚
            self.reward_type = "slope" # reward的类型
        self.dis_p_n=args.dismantling_number # 瓦解目标规模
        self.path = path # 瓦解数据存储路径

        self.seed=None  # 生成的网络种子
        self.Graph,self.pos = None,None
        self.G = None
        self.Rc_list = None
        self.curve_list = None
        self.average_degree = None

        # action_dict = dict()  # 边对应的action index  # 1:一步生成边序号  2:分两步依次生成两个节点   3:一步生成两个节点
        # e = 0
        # for i in range(0, self.n - 1):
        #     for j in range(i + 1, self.n):
        #         action_dict[e] = [i, j]
        #         e += 1
        # self.action_dict = action_dict

    def reset(self,seed,new_graph_type=0,infer=False): # 可修改seed来修改初始树结构
        if new_graph_type!=0: self.graph_type=new_graph_type # 如果reset的网络类型改变，则改变self网络类型属性
        self.seed = seed
        self.Graph,self.pos = generate_network(self.n,self.graph_type,seed)
        self.average_degree = len(self.Graph.edges())/len(self.Graph)*2
        self.G = copy.deepcopy(self.Graph)
        if infer==False:
            R, curve, method = dismantle(self.dismantling_name,self.G,self.dis_p_n,self.path)
        else:
            R, curve, method=0,0,0
        self.Rc_list = [R]
        self.curve_list = [curve]
        return copy.deepcopy(self.G),self.pos  # s0,pos

    def step(self,act): # act:表示增加一条边，动作空间为n*(n-1)-(n-1)-t 总连边数
        if (act[0],act[1]) in self.G.edges():
            if self.penalty:
                r=self.penalty
            else:
                r=0
        else:
            if act[0]==act[1]:
                print(1)
            self.G.add_edge(act[0],act[1])
            r = self.reward()
        done=0
        # if len(self.Graph.edges())>120:
        if self.average_degree>3 or self.n>200: # 平均度为4
            stop = len(self.Graph.edges())+100 # 初始边数+100
        else: #平均度为2
            stop = len(self.Graph)+100 # 节点数+100
        if len(self.G.edges())>=stop: done=1
        return r,copy.deepcopy(self.G),done  # r,s_,done

    def reward(self):
        R, curve, method = dismantle(self.dismantling_name,self.G,self.dis_p_n,self.path)
        self.Rc_list.append(R)
        self.curve_list.append(curve)
        if self.reward_type=="slope":  r = self.Rc_list[-1]-self.Rc_list[-2] #最大化最终鲁棒性值
        if self.reward_type=="area1":  r = self.Rc_list[-1]/len(self.G)  # Finder的结果:最大化面积
        if self.reward_type=="area2":  r = (self.Rc_list[-1]+self.Rc_list[-2])/(2*len(self.G))  # tc师兄：最大化面积
        if self.reward_type=="area3":  r  = sum(self.Rc_list)/len(self.Rc_list)- sum(self.Rc_list[:-1])/(len(self.Rc_list[:-1])*len(self.G)) # tc师兄
        return r


class cost_env(object): # random_cost
    def __init__(self, args,path=0):
        self.n = args.num_nodes # 网络规模
        self.dismantling_name = args.dismantling_name # 瓦解方法
        self.graph_type = args.graph_type # 生成的网络类型
        self.penalty=args.env_penalty # 是否有添加边失败的惩罚
        self.dis_p_n=args.dismantling_number # 瓦解目标规模
        self.reward_type = args.reward_type # reward的类型
        self.path = path # 瓦解数据存储路径

        self.sum_cost = args.cost # 连边总成本
        self.now_cost = self.sum_cost

        self.seed=None  # 生成的网络种子
        self.Graph,self.pos = None,None
        self.G = None
        self.Rc_list = None
        self.curve_list = None
        self.average_degree=None
        self.node_cost = None

        action_dict = dict()  # 边对应的action index  # 1:一步生成边序号  2:分两步依次生成两个节点   3:一步生成两个节点
        e = 0
        for i in range(0, self.n - 1):
            for j in range(i + 1, self.n):
                action_dict[e] = [i, j]
                e += 1
        self.action_dict = action_dict

    def reset(self,seed,new_graph_type=0): # 可修改seed来修改初始树结构
        self.now_cost = self.sum_cost
        if new_graph_type!=0: self.graph_type=new_graph_type # 如果reset的网络类型改变，则改变self网络类型属性
        self.seed = seed
        self.Graph,self.pos,self.node_cost = generate_network_cost(self.n,self.graph_type,seed)
        self.average_degree = len(self.Graph.edges())/len(self.Graph)*2
        self.G = copy.deepcopy(self.Graph)
        R, curve, method = dismantle(self.dismantling_name,self.G,self.dis_p_n,self.path)
        self.Rc_list = [R]
        self.curve_list = [curve]
        return copy.deepcopy(self.G),self.now_cost,self.node_cost

    def step(self,act): # act:表示增加一条边，动作空间为n*(n-1)-(n-1)-t 总连边数
        # self.now_cost -= (self.G.degree(act[0])+self.G.degree(act[1]))
        self.now_cost -= (self.node_cost[act[0]]+self.node_cost[act[1]])
        self.G.add_edge(act[0],act[1])
        r = self.reward()

        # done = 0
        return r,copy.deepcopy(self.G),self.now_cost  # r,s_,done

    def reward(self):
        R, curve, method = dismantle(self.dismantling_name,self.G,self.dis_p_n,self.path)
        self.Rc_list.append(R)
        self.curve_list.append(curve)
        if self.reward_type=="slope":  r = self.Rc_list[-1]-self.Rc_list[-2] #最大化最终鲁棒性值
        if self.reward_type=="area1":  r = self.Rc_list[-1]/len(self.G)  # Finder的结果:最大化面积
        if self.reward_type=="area2":  r = (self.Rc_list[-1]+self.Rc_list[-2])/(2*len(self.G))  # tc师兄：最大化面积
        if self.reward_type=="area3":  r  = sum(self.Rc_list)/len(self.Rc_list)- sum(self.Rc_list[:-1])/(len(self.Rc_list[:-1])*len(self.G)) # tc师兄
        # if r<0:
        #     print(1)
        return r
