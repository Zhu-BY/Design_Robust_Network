import subprocess

# 步骤 1: 生成图
with open('graph.txt', 'w') as f:
    subprocess.run(['python', 'gnp.py', '78125', '3.5', '1'], stdout=f)

# 步骤 2: 找到去环集合
with open('graph.txt', 'r') as graph_file, open('seeds.txt', 'w') as seeds_file:
    subprocess.run(['./decycler', '-o'], stdin=graph_file, stdout=seeds_file)

# 步骤 3: 将去环图分解为大小 <= 100 的组件
with open('graph.txt', 'r') as graph_file, open('seeds.txt', 'r') as seeds_file, open('broken.txt', 'w') as broken_file:
    subprocess.run(['type'], stdin=graph_file, stdout=subprocess.PIPE)
    subprocess.run(['type'], stdin=seeds_file, stdout=subprocess.PIPE)
    subprocess.run(['python', 'treebreaker.py', '100'], stdin=subprocess.PIPE, stdout=broken_file)

# 步骤 4: 重新引入被移除的节点，直到组件的大小 <= 200
with open('graph.txt', 'r') as graph_file, open('seeds.txt', 'r') as seeds_file, open('broken.txt', 'r') as broken_file, open(
        'output.txt', 'w') as output_file:
    subprocess.run(['type'], stdin=graph_file, stdout=subprocess.PIPE)
    subprocess.run(['type'], stdin=seeds_file, stdout=subprocess.PIPE)
    subprocess.run(['type'], stdin=broken_file, stdout=subprocess.PIPE)
    subprocess.run(['./reverse-greedy', '-t', '200'], stdin=subprocess.PIPE, stdout=output_file)

# 输出结果
with open('output.txt', 'r') as f:
    seeds = f.read()
    print(seeds)
