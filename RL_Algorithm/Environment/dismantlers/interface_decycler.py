import copy
import subprocess
import copy
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx as nx
def min_sum_with_seeds(G,size1=4,size2=4,txtpath=-1):
    # G = nx.gnm_random_graph(2000,4000)
    if G==0:
        # G= nx.random_regular_graph(4,100)
        n=100
        G = nx.erdos_renyi_graph(100,0.01,seed=0)
    path = "E:\\CSR\\b\\Optimal_Graph_Generation\\dismantlers\\decycler-master\\"
    gnp_script_path = path+'gnp.py'
    decycler_executable_path = path + 'decycler'
    treebreaker_script_path = path + 'treebreaker.py'
    reverse_greedy_executable_path = path + 'reverse-greedy'
    if txtpath==-1:txtpath=path # 没有传入路径数据时，使用本地路径
    elif isinstance(txtpath, int): # 如何传入路径参数为一个整数
        txtpath = path+'data\\'+str(txtpath)
    graph_txt_path = txtpath + 'graph.txt'
    seeds_txt_path = txtpath + 'seeds.txt'
    broken_txt_path = txtpath + 'broken.txt'
    output_txt_path = txtpath + 'output.txt'

    # 步骤 1: 生成图
    try:
        with open(graph_txt_path, 'w',encoding='utf-8') as file:
            for edge in G.edges():
                file.write("D {} {}\n".format(edge[0], edge[1]))
    except:
        print('error')
    # print(f"边已保存到文件 {graph_txt_path}")
    #
    # with open(graph_txt_path, 'w') as f:
    #     subprocess.run(['python', gnp_script_path, '78125', '3.5', '1'], stdout=f)

    # 步骤 2: 找到去环集合
    try:
        with open(graph_txt_path, 'r',encoding='utf-8') as graph_file, open(seeds_txt_path, 'w',encoding='utf-8') as seeds_file:
            subprocess.run([decycler_executable_path, '-o'], stdin=graph_file, stdout=seeds_file)
    except:
        print('error')
    # 步骤 3: 将去环图分解为大小 <= 100 的组件
    # 方法 1-1
    try:
        with open(graph_txt_path, 'r',encoding='utf-8') as graph_file, open(seeds_txt_path, 'r',encoding='utf-8') as seeds_file:
            input_text = graph_file.read() + seeds_file.read()
        process = subprocess.run(['python', treebreaker_script_path, str(size1)], input=input_text, text=True,capture_output=True, encoding='utf-8')
        with open(broken_txt_path, 'w',encoding='utf-8') as broken_file:
            broken_file.write(process.stdout)
    except:
        print('error')

    # 方法 1-2
    # with open(broken_txt_path, 'w') as broken_file:
    #     # 先读取 graph_txt_path 和 seeds_txt_path 的内容
    #     with open(graph_txt_path, 'r') as graph_file, open(seeds_txt_path, 'r') as seeds_file:
    #         graph_content = graph_file.read()
    #         seeds_content = seeds_file.read()
    #     # 通过 subprocess.run 将内容传递给 treebreaker_script_path
    #     subprocess.run(['python', treebreaker_script_path, str(size1)], input=graph_content + seeds_content, text=True, stdout=broken_file)
   # 方法 2
   #  command = f'type {graph_txt_path} {seeds_txt_path}'
   #  with open(broken_txt_path, 'w') as broken_file:
   #      process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
   #      subprocess.run(['python', treebreaker_script_path, str(size1)], stdin=process.stdout, stdout=broken_file)

    # 步骤 4: 重新引入被移除的节点，直到组件的大小 <= 200
    # 方法 1
    try:
        with open(output_txt_path, 'w',encoding='utf-8') as output_file:
            with open(graph_txt_path, 'r',encoding='utf-8') as graph_file, open(seeds_txt_path, 'r',encoding='utf-8') as seeds_file, open(broken_txt_path,'r',encoding='utf-8') as broken_file:
                graph_content = graph_file.read()
                seeds_content = seeds_file.read()
                broken_content = broken_file.read()
            subprocess.run([reverse_greedy_executable_path, '-t', str(size2)],
                                     input=graph_content + seeds_content + broken_content, text=True,
                                     stdout=output_file, encoding='utf-8')
    except:
        print('error')
    #方法 2
    # command = f'type {graph_txt_path} {seeds_txt_path} {broken_txt_path}'
    # with open(output_txt_path, 'w') as output_file:
    #     process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    #     subprocess.run([reverse_greedy_executable_path, '-t', str(size2)], stdin=process.stdout, stdout=output_file)
    # 步骤 5：导入output结果output_txt_path，提取seed，计算R

    seed_numbers = []
    seeds_section = True
    with open(output_txt_path, 'r',encoding='utf-8') as file:
        for line in file:
            # # Check if we've reached the seeds section
            # if line.strip() == "# Seeds":
            #     seeds_section = True
            #     continue
            # # If in seeds section and line starts with 'S', extract the number
            if seeds_section and line.startswith("S "):
                parts = line.split()
                if len(parts) == 2:
                    seed_numbers.append(int(parts[1]))
                continue

            # If we've reached another section, stop reading further
            # if seeds_section and not line.startswith("S "):
            #     break
    # print(seed_numbers)
    # 读取图
    G = nx.Graph()
    # 打开并读取文件
    with open(graph_txt_path, 'r',encoding='utf-8') as file:
        for line in file:
            # 忽略以D开头的字符，并分割剩余部分
            _, node1, node2 = line.split()
            # 添加边到图中，节点类型根据需要自行转换（这里假设为整数）
            G.add_edge(int(node1), int(node2))
    # 打印图信息，验证导入是否成功
    lcc_list = [len(G)]
    G_=copy.deepcopy(G)
    for seed in seed_numbers:
        G_.remove_node(seed)
        lcc = len(max(list(nx.connected_components(G_)),key=len))
        lcc_list.append(lcc)
        # print(lcc)
    # plt.plot(lcc_list)
    # plt.show()
    # print("Robustness:",sum(lcc_list)/len(G))
    if len(G)!=0:
        return sum(lcc_list)/len(G),[x/len(G) for x in lcc_list],seed_numbers
    else:
        return 0,[0],seed_numbers
def min_sum(G,size1=4,size2=4,txtpath=-1):
    # G = nx.gnm_random_graph(2000,4000)
    if G==0:
        # G= nx.random_regular_graph(4,100)
        n=100
        G = nx.erdos_renyi_graph(100,0.01,seed=0)
    path = "E:\\CSR\\b\\Optimal_Graph_Generation\\dismantlers\\decycler-master\\"
    gnp_script_path = path+'gnp.py'
    decycler_executable_path = path + 'decycler'
    treebreaker_script_path = path + 'treebreaker.py'
    reverse_greedy_executable_path = path + 'reverse-greedy'
    if txtpath==-1:txtpath=path # 没有传入路径数据时，使用本地路径
    elif isinstance(txtpath, int): # 如何传入路径参数为一个整数
        txtpath = path+'data\\'+str(txtpath)
    graph_txt_path = txtpath + 'graph.txt'
    seeds_txt_path = txtpath + 'seeds.txt'
    broken_txt_path = txtpath + 'broken.txt'
    output_txt_path = txtpath + 'output.txt'

    # 步骤 1: 生成图
    try:
        with open(graph_txt_path, 'w',encoding='utf-8') as file:
            for edge in G.edges():
                file.write("D {} {}\n".format(edge[0], edge[1]))
    except:
        print('error')
    # print(f"边已保存到文件 {graph_txt_path}")
    #
    # with open(graph_txt_path, 'w') as f:
    #     subprocess.run(['python', gnp_script_path, '78125', '3.5', '1'], stdout=f)

    # 步骤 2: 找到去环集合
    try:
        with open(graph_txt_path, 'r',encoding='utf-8') as graph_file, open(seeds_txt_path, 'w',encoding='utf-8') as seeds_file:
            subprocess.run([decycler_executable_path, '-o'], stdin=graph_file, stdout=seeds_file)
    except:
        print('error')
    # 步骤 3: 将去环图分解为大小 <= 100 的组件
    # 方法 1-1
    try:
        with open(graph_txt_path, 'r',encoding='utf-8') as graph_file, open(seeds_txt_path, 'r',encoding='utf-8') as seeds_file:
            input_text = graph_file.read() + seeds_file.read()
        process = subprocess.run(['python', treebreaker_script_path, str(size1)], input=input_text, text=True,capture_output=True, encoding='utf-8')
        with open(broken_txt_path, 'w',encoding='utf-8') as broken_file:
            broken_file.write(process.stdout)
    except:
        print('error')

    # 方法 1-2
    # with open(broken_txt_path, 'w') as broken_file:
    #     # 先读取 graph_txt_path 和 seeds_txt_path 的内容
    #     with open(graph_txt_path, 'r') as graph_file, open(seeds_txt_path, 'r') as seeds_file:
    #         graph_content = graph_file.read()
    #         seeds_content = seeds_file.read()
    #     # 通过 subprocess.run 将内容传递给 treebreaker_script_path
    #     subprocess.run(['python', treebreaker_script_path, str(size1)], input=graph_content + seeds_content, text=True, stdout=broken_file)
   # 方法 2
   #  command = f'type {graph_txt_path} {seeds_txt_path}'
   #  with open(broken_txt_path, 'w') as broken_file:
   #      process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
   #      subprocess.run(['python', treebreaker_script_path, str(size1)], stdin=process.stdout, stdout=broken_file)

    # 步骤 4: 重新引入被移除的节点，直到组件的大小 <= 200
    # 方法 1
    try:
        with open(output_txt_path, 'w',encoding='utf-8') as output_file:
            with open(graph_txt_path, 'r',encoding='utf-8') as graph_file, open(seeds_txt_path, 'r',encoding='utf-8') as seeds_file, open(broken_txt_path,'r',encoding='utf-8') as broken_file:
                graph_content = graph_file.read()
                seeds_content = seeds_file.read()
                broken_content = broken_file.read()
            subprocess.run([reverse_greedy_executable_path, '-t', str(size2)],
                                     input=graph_content + seeds_content + broken_content, text=True,
                                     stdout=output_file, encoding='utf-8')
    except:
        print('error')
    #方法 2
    # command = f'type {graph_txt_path} {seeds_txt_path} {broken_txt_path}'
    # with open(output_txt_path, 'w') as output_file:
    #     process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    #     subprocess.run([reverse_greedy_executable_path, '-t', str(size2)], stdin=process.stdout, stdout=output_file)
    # 步骤 5：导入output结果output_txt_path，提取seed，计算R

    seed_numbers = []
    seeds_section = True
    with open(output_txt_path, 'r',encoding='utf-8') as file:
        for line in file:
            # # Check if we've reached the seeds section
            # if line.strip() == "# Seeds":
            #     seeds_section = True
            #     continue
            # # If in seeds section and line starts with 'S', extract the number
            if seeds_section and line.startswith("S "):
                parts = line.split()
                if len(parts) == 2:
                    seed_numbers.append(int(parts[1]))
                continue

            # If we've reached another section, stop reading further
            # if seeds_section and not line.startswith("S "):
            #     break
    # print(seed_numbers)
    # 读取图
    G = nx.Graph()
    # 打开并读取文件
    with open(graph_txt_path, 'r',encoding='utf-8') as file:
        for line in file:
            # 忽略以D开头的字符，并分割剩余部分
            _, node1, node2 = line.split()
            # 添加边到图中，节点类型根据需要自行转换（这里假设为整数）
            G.add_edge(int(node1), int(node2))
    # 打印图信息，验证导入是否成功
    lcc_list = [len(G)]
    G_=copy.deepcopy(G)
    for seed in seed_numbers:
        G_.remove_node(seed)
        lcc = len(max(list(nx.connected_components(G_)),key=len))
        lcc_list.append(lcc)
        # print(lcc)
    # plt.plot(lcc_list)
    # plt.show()
    # print("Robustness:",sum(lcc_list)/len(G))
    if len(G)!=0:
        return sum(lcc_list)/len(G),[x/len(G) for x in lcc_list]
    else:
        return 0,[0]
if __name__=="__main__":

    min_sum(0,1,1)
    # 设定各个文件和脚本的路径
    path="E:\\CSR\\b\\Optimal_Graph_Generation\\dismantlers\\decycler-master\\"
    gnp_script_path = path+'gnp.py'
    decycler_executable_path = path+'decycler'
    treebreaker_script_path = path+'treebreaker.py'
    reverse_greedy_executable_path = path+'reverse-greedy'

    graph_txt_path = path+'graph.txt'
    seeds_txt_path = path+'seeds.txt'
    broken_txt_path = path+'broken.txt'
    output_txt_path = path+'output.txt'

    # 步骤 1: 生成图
    with open(graph_txt_path, 'w') as f:
        subprocess.run(['python', gnp_script_path, '78125', '3.5', '1'], stdout=f)

    # 步骤 2: 找到去环集合
    with open(graph_txt_path, 'r') as graph_file, open(seeds_txt_path, 'w') as seeds_file:
        subprocess.run([decycler_executable_path, '-o'], stdin=graph_file, stdout=seeds_file)

    # 步骤 3: 将去环图分解为大小 <= 100 的组件
    # with open(graph_txt_path, 'r') as graph_file, open(seeds_txt_path, 'r') as seeds_file, open(broken_txt_path, 'w') as broken_file:
    #     # Windows 中的 'type' 命令等同于 Unix/Linux 中的 'cat'
    #     subprocess.run(['type', graph_txt_path], shell=True, stdout=broken_file)
    #     subprocess.run(['type', seeds_txt_path], shell=True, stdout=broken_file)
    #     subprocess.run(['python', treebreaker_script_path, '100'], stdin=subprocess.PIPE, stdout=broken_file)
    # with open(graph_txt_path, 'r') as graph_file, open(seeds_txt_path, 'w') as seeds_file:
    #     subprocess.run([decycler_executable_path, '-o'], stdin=graph_file, stdout=seeds_file)

    # 步骤 3: 将去环图分解为大小 <= 100 的组件
    # with open(graph_txt_path, 'r') as graph_file, open(seeds_txt_path, 'r') as seeds_file:
    #     input_text = graph_file.read() + "\n" + seeds_file.read()
    # process = subprocess.run(['python', treebreaker_script_path, '10'], input=input_text, text=True,capture_output=True)
    # with open(broken_txt_path, 'w') as broken_file:
    #     broken_file.write(process.stdout)
    command = f'type {graph_txt_path} {seeds_txt_path}'
    with open(broken_txt_path, 'w') as broken_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        subprocess.run(['python', treebreaker_script_path, '1'], stdin=process.stdout, stdout=broken_file)

    # 步骤 4: 重新引入被移除的节点，直到组件的大小 <= 200
    # with open(graph_txt_path, 'r') as graph_file, open(seeds_txt_path, 'r') as seeds_file, open(broken_txt_path,
    #                                                                                             'r') as broken_file, open(
    #         output_txt_path, 'w') as output_file:
    #     subprocess.run(['type', graph_txt_path], shell=True, stdout=output_file)
    #     subprocess.run(['type', seeds_txt_path], shell=True, stdout=output_file)
    #     subprocess.run(['type', broken_txt_path], shell=True, stdout=output_file)
    #     subprocess.run([reverse_greedy_executable_path, '-t', '10'], stdin=subprocess.PIPE, stdout=output_file)
    command = f'type {graph_txt_path} {seeds_txt_path} {broken_txt_path}'
    with open(output_txt_path, 'w') as output_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        subprocess.run([reverse_greedy_executable_path, '-t', '1'], stdin=process.stdout, stdout=output_file)
    # 步骤 5：导入output结果output_txt_path，提取seed，计算R
    seed_numbers = []
    seeds_section = True
    with open(output_txt_path, 'r') as file:
        for line in file:
            # # Check if we've reached the seeds section
            # if line.strip() == "# Seeds":
            #     seeds_section = True
            #     continue
            # # If in seeds section and line starts with 'S', extract the number
            if seeds_section and line.startswith("S "):
                parts = line.split()
                if len(parts) == 2:
                    seed_numbers.append(int(parts[1]))
                continue

            # If we've reached another section, stop reading further
            if seeds_section and not line.startswith("S "):
                break
    print(seed_numbers)
    # 读取图
    G = nx.Graph()
    # 打开并读取文件
    with open(path + 'graph.txt', 'r') as file:
        for line in file:
            # 忽略以D开头的字符，并分割剩余部分
            _, node1, node2 = line.split()
            # 添加边到图中，节点类型根据需要自行转换（这里假设为整数）
            G.add_edge(int(node1), int(node2))
    # 打印图信息，验证导入是否成功
    lcc_list = []
    for seed in seed_numbers:
        G.remove_node(seed)
        lcc = len(max(list(nx.connected_components(G)), key=len))
        lcc_list.append(lcc)
        print(lcc)
    plt.plot(lcc_list)
    plt.show()
    print("Robustness:", sum(lcc_list) / len(G))

