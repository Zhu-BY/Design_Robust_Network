import networkx as nx
from RL_Algorithm.Environment.dismantlers.interface_decycler import min_sum_with_seeds
from RL_Algorithm.Environment.dismantlers.interface_gnd import gnd_with_seeds
from RL_Algorithm.Environment.dismantlers.heuristic_dismantler import HBA_with_seeds,HDA_with_seeds,CI2_with_seeds
import warnings
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

def dismantle_with_seeds(method,G0,target_size=4,txtpath=-1):
    G = nx.convert_node_labels_to_integers(G0)
    if method == 'HBA':
        R,r_list,remove_seeds = HBA_with_seeds(G,target_size)
        return R, r_list, method,remove_seeds
    if method == 'HDA':
        R,r_list,remove_seeds = HDA_with_seeds(G,target_size)
        return R, r_list, method,remove_seeds
    if method == 'CI2':
        R,r_list,remove_seeds = CI2_with_seeds(G,target_size,2)
        return R, r_list, method,remove_seeds
    if method == 'MS':
        R,r_list,remove_seeds = min_sum_with_seeds(G,target_size,target_size,txtpath=txtpath)
        return R, r_list, method,remove_seeds
    if method == 'GND':
        R,r_list,remove_seeds = gnd_with_seeds(G,target_size,R=0,txtpath=txtpath)
        return R, r_list, method,remove_seeds
    if method == 'GNDR':
        R,r_list,remove_seeds = gnd_with_seeds(G,target_size,R=1,txtpath=txtpath)
        return R, r_list, method,remove_seeds

if __name__=='__main__':
    n=1000
    G = nx.barabasi_albert_graph(n, 4)
    r_list_list = []
    for method in ["RB","RD","CI1","MS","GND","GNDR",'CoreHD']:
        print(method)
        R,r_list,method = dismantle_with_seeds(method,G,1)
        plt.plot(r_list,label=method)
        r_list_list.append(r_list)
    plt.legend()
    plt.savefig('dismantling_curve of all attacks.jpg')
    plt.show()
    print(method)
    print('finish')