import networkx as nx
from RL_Algorithm.Environment.dismantlers.interface_decycler import min_sum
from RL_Algorithm.Environment.dismantlers.interface_gnd import gnd
from RL_Algorithm.Environment.dismantlers.heuristic_dismantler import HDA,HBA,CI2
from RL_Algorithm.Environment.dismantlers.mix_dismantle import mix_dis
from RL_Algorithm.Environment.dismantlers.interface_NIRM import nirm
import warnings
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
try:
    from RL_Algorithm.Environment.dismantlers.interface_GDM_BA import gdm_ba
    from RL_Algorithm.Environment.dismantlers.interface_GDM_ER import gdm_er
except:
    print('GDM is valid')
    def gdm_ba(a=0,b=0):
        return a,b
    def gdm_er(a=0,b=0):
        return a,b
def dismantle(method,G0,target_size=1,txtpath=-1,graph_type='BA'):
    """
    :param G: networkx格式的网络
    :param target_size: 瓦解过程停止条件，即网络最大连通子团大小，默认为4（GND在小于4时可能失效）
    :param method: 瓦解方法
    :return: 对应方法下的鲁棒性指标与瓦解曲线
    """
    G = nx.convert_node_labels_to_integers(G0)
    if method == 'GDM':
        if type(txtpath)==str:
            txtpath=txtpath[-3]+'rl'
        if graph_type=='BA':
            R, r_list = gdm_ba(G, txtpath)
        else:
            R,r_list = gdm_er(G,txtpath)
        return R,r_list,method
    if method == 'NIRM':
        R,r_list = nirm(G,-99)
        return R,r_list,method
    if method == 'HBA':
        R,r_list = HBA(G,target_size)
        return R, r_list, method
    if method == 'HDA':
        R,r_list = HDA(G,target_size)
        return R, r_list, method
    if method == 'CI2':
        R,r_list = CI2(G,target_size,2)
        return R, r_list, method
    if method == 'MS':
        R,r_list = min_sum(G,target_size,target_size,txtpath=txtpath)
        return R, r_list, method
    if method == 'GND':
        target_size=4
        R,r_list = gnd(G,target_size,R=0,txtpath=txtpath)
        return R, r_list, method
    if method == 'GNDR':
        target_size=4
        R,r_list = gnd(G,target_size,R=1,txtpath=txtpath)
        return R, r_list, method
    if method=='Mix':
        R,r_list = mix_dis(G)
        return R, r_list, method
if __name__=='__main__':
    n=1000
    G = nx.barabasi_albert_graph(n, 4)
    r_list_list = []
    for method in ["RB","RD","CI1","MS","GND","GNDR",'CoreHD']:
        print(method)
        R,r_list,method = dismantle(method,G,1)
        plt.plot(r_list,label=method)
        r_list_list.append(r_list)
    plt.legend()
    # plt.xlim(0,250)
    plt.savefig('dismantling_curve of all attacks.jpg')
    plt.show()
    print(method)
    print('finish')