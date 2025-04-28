import copy
import subprocess
import networkx as nx
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
def gnd_with_seeds(G,target_size=4,R=0,size2=4,txtpath=-1):
    # G = nx.barabasi_albert_graph(100,2)
    if G==0:
        G = nx.random_regular_graph(4,100)
        n=100
        G = nx.erdos_renyi_graph(100,0.01,seed=0)
    if nx.is_connected(G)==False:
        for node in list(G.nodes):
            if G.degree(node)==0:
                G.remove_node(node)
        # Gcc = max(list(nx.connected_components(G)),key=len)
        # G = G.subgraph(Gcc)
        G = nx.convert_node_labels_to_integers(G)
    # 设置你的GND程序的路径和所有必要的命令行参数
    path = "E:/CSR/b/Optimal_Graph_Generation/dismantlers/Generalized-Network-Dismantling-Input/"
    gnd_program_path = path+"GND.exe"
    if txtpath==-1:txtpath=path # 没有传入路径数据时，使用本地路径
    elif isinstance(txtpath, int): # 如何传入路径参数为一个整数
        txtpath = path+'data/'+str(txtpath)
    file_net = txtpath+"Graph.txt"
    file_id = txtpath+"NodeSet_GND_weighted_Graph.txt"
    file_plot =txtpath+"Plot_GND_weighted_Graph.txt"
    node_num = str(len(G))
    remove_strategy = "3" #1: weighted method, 3: unweighted method
    plot_size = "1"
    if len(G)==0:
        return 0,[0]
    target_size_ratio = str(target_size/len(G))
    # 生成gnd可以处理的网络格式
    with open(file_net, 'w') as file:
        for edge in G.edges():
            file.write("{} {}\n".format(edge[0]+1, edge[1]+1)) # GND中的图的节点id要求大于1
    # print(f"边已保存到文件 {file_net}")
    # 构造命令行命令
    command = [
        gnd_program_path,
        node_num,
        file_net,
        file_id,
        file_plot,
        remove_strategy,
        plot_size,
        '0.01'
    ]
    # 调用GND程序
    result = subprocess.run(command, capture_output=True, text=True)

    # 打印命令行输出（如果有的话）
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)
    if R==1:
        # 假设'reinsertion'是另一个可执行文件，首先指定它的完整路径
        reinsertion_program_path = path+"reinsertion.exe"

        # 设置命令行参数
        arguments = [
            "-t", str(size2),
            "-N", txtpath+"Graph.txt",
            "-I", txtpath+"NodeSet_GND_weighted_Graph.txt",
            "-D", txtpath+"NodeSet_GNDR_weighted_Graph.txt",
            "-S", "2" # 0: keep the original order
                  #  1: ascending order - better strategy for weighted case
            #2: descending order - better strategy for unweighted case
        ]
        # 构造完整的命令行命令
        command = [reinsertion_program_path] + arguments
        # 调用reinsertion程序
        result = subprocess.run(command, capture_output=True, text=True)
        # 打印命令行输出
        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)

    # 读取结果瓦解节点结合
    if R!=1: file =txtpath+"NodeSet_GND_weighted_Graph.txt"
    if R==1:file =txtpath+"NodeSet_GNDR_weighted_Graph.txt"
    seed_numbers = []

    with open(file, 'r') as file:
        for line in file:
            seed_numbers.append(int(line))
    # print(seed_numbers)
    # 读取图
    G = nx.Graph()
    # 打开并读取文件
    with open(txtpath+'Graph.txt', 'r') as file:
        for line in file:
            # 忽略以D开头的字符，并分割剩余部分
            node1, node2 = line.split()
            # 添加边到图中，节点类型根据需要自行转换（这里假设为整数）
            G.add_edge(int(node1), int(node2))
    # 绘制瓦解曲线
    lcc_list = [len(G)]
    G_=copy.deepcopy(G)
    for seed in seed_numbers:
        try:
            G_.remove_node(seed)
            lcc = len(max(list(nx.connected_components(G_)), key=len))
            lcc_list.append(lcc)
        except:
            print(1)
    # plt.plot(lcc_list)
    # plt.show()
    # print("Robustness:",sum(lcc_list)/len(G))
    if seed_numbers==[]:
        print(1)
    return sum(lcc_list)/len(G),[x/len(G) for x in lcc_list],[x-1 for x in seed_numbers]
def gnd(G,target_size=4,R=0,size2=4,txtpath=-1):
    # G = nx.barabasi_albert_graph(100,2)
    if G==0:
        G = nx.random_regular_graph(4,100)
        n=100
        G = nx.erdos_renyi_graph(100,0.01,seed=0)
    if nx.is_connected(G)==False:
        for node in list(G.nodes):
            if G.degree(node)==0:
                G.remove_node(node)
        # Gcc = max(list(nx.connected_components(G)),key=len)
        # G = G.subgraph(Gcc)
        G = nx.convert_node_labels_to_integers(G)
    # 设置你的GND程序的路径和所有必要的命令行参数
    path = "E:/CSR/b/Optimal_Graph_Generation/dismantlers/Generalized-Network-Dismantling-Input/"
    gnd_program_path = path+"GND.exe"
    if txtpath==-1:txtpath=path # 没有传入路径数据时，使用本地路径
    elif isinstance(txtpath, int): # 如何传入路径参数为一个整数
        txtpath = path+'data/'+str(txtpath)
    file_net = txtpath+"Graph.txt"
    file_id = txtpath+"NodeSet_GND_weighted_Graph.txt"
    file_plot =txtpath+"Plot_GND_weighted_Graph.txt"
    node_num = str(len(G))
    remove_strategy = "3" #1: weighted method, 3: unweighted method
    plot_size = "1"
    if len(G)==0:
        return 0,[0]
    target_size_ratio = str(target_size/len(G))
    # 生成gnd可以处理的网络格式
    with open(file_net, 'w') as file:
        for edge in G.edges():
            file.write("{} {}\n".format(edge[0]+1, edge[1]+1)) # GND中的图的节点id要求大于1
    # print(f"边已保存到文件 {file_net}")
    # 构造命令行命令
    command = [
        gnd_program_path,
        node_num,
        file_net,
        file_id,
        file_plot,
        remove_strategy,
        plot_size,
        target_size_ratio
    ]
    # 调用GND程序
    result = subprocess.run(command, capture_output=True, text=True)

    # 打印命令行输出（如果有的话）
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)
    if R==1:
        # 假设'reinsertion'是另一个可执行文件，首先指定它的完整路径
        reinsertion_program_path = path+"reinsertion.exe"

        # 设置命令行参数
        arguments = [
            "-t", str(size2),
            "-N", txtpath+"Graph.txt",
            "-I", txtpath+"NodeSet_GND_weighted_Graph.txt",
            "-D", txtpath+"NodeSet_GNDR_weighted_Graph.txt",
            "-S", "2" # 0: keep the original order
                  #  1: ascending order - better strategy for weighted case
            #2: descending order - better strategy for unweighted case
        ]
        # 构造完整的命令行命令
        command = [reinsertion_program_path] + arguments
        # 调用reinsertion程序
        result = subprocess.run(command, capture_output=True, text=True)
        # 打印命令行输出
        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)

    # 读取结果瓦解节点结合
    if R!=1: file =txtpath+"NodeSet_GND_weighted_Graph.txt"
    if R==1:file =txtpath+"NodeSet_GNDR_weighted_Graph.txt"
    seed_numbers = []

    with open(file, 'r') as file:
        for line in file:
            seed_numbers.append(int(line))
    # print(seed_numbers)
    # 读取图
    G = nx.Graph()
    # 打开并读取文件
    with open(txtpath+'Graph.txt', 'r') as file:
        for line in file:
            # 忽略以D开头的字符，并分割剩余部分
            node1, node2 = line.split()
            # 添加边到图中，节点类型根据需要自行转换（这里假设为整数）
            G.add_edge(int(node1), int(node2))
    # 绘制瓦解曲线
    lcc_list = [len(G)]
    G_=copy.deepcopy(G)
    for seed in seed_numbers:
        try:
            G_.remove_node(seed)
            lcc = len(max(list(nx.connected_components(G_)), key=len))
            lcc_list.append(lcc)
        except:
            print(1)
    # plt.plot(lcc_list)
    # plt.show()
    # print("Robustness:",sum(lcc_list)/len(G))
    return sum(lcc_list)/len(G),[x/len(G) for x in lcc_list]
if __name__=="__main__":
    # path = "E:/CSR/b/Optimal_Graph_Generation/dismantlers/Generalized-Network-Dismantling-Input/"
    # G = nx.Graph()
    # # 打开并读取文件
    # with open(path+'Graph.txt', 'r') as file:
    #     for line in file:
    #         # 忽略以D开头的字符，并分割剩余部分
    #         node1, node2 = line.split()
    #         # 添加边到图中，节点类型根据需要自行转换（这里假设为整数）
    #         G.add_edge(int(node1), int(node2))

    gnd(0,1,R=0)

    # 设置你的GND程序的路径和所有必要的命令行参数
    path = "E:/CSR/b/Optimal_Graph_Generation/dismantlers/Generalized-Network-Dismantling-Input/"
    gnd_program_path = path+"GND.exe"
    node_num = "754"
    file_net = path+"CrimeNet.txt"
    file_id = path+"NodeSet_GND_weighted_CrimeNet.txt"
    file_plot =path+"Plot_GND_weighted_CrimeNet.txt"
    remove_strategy = "3"
    plot_size = "1"
    target_size_ratio = "0.0053"

    # 构造命令行命令
    command = [
        gnd_program_path,
        node_num,
        file_net,
        file_id,
        file_plot,
        remove_strategy,
        plot_size,
        target_size_ratio
    ]

    # 调用GND程序
    result = subprocess.run(command, capture_output=True, text=True)

    # 打印命令行输出（如果有的话）
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)


    R=1
    if R==1:
        # 假设'reinsertion'是另一个可执行文件，首先指定它的完整路径
        reinsertion_program_path = path+"reinsertion.exe"

        # 设置命令行参数
        arguments = [
            "-t", "7",
            "-N", path+"CrimeNet.txt",
            "-I", path+"NodeSet_GND_weighted_CrimeNet.txt",
            "-D", path+"NodeSet_GNDR_weighted_CrimeNet.txt",
            "-S", "1"
        ]

        # 构造完整的命令行命令
        command = [reinsertion_program_path] + arguments

        # 调用reinsertion程序
        result = subprocess.run(command, capture_output=True, text=True)

        # 打印命令行输出
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
