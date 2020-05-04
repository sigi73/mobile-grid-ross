fn = 'gen_execute.sh'
f = open(fn, 'w')


mkdir_format_string = 'mkdir -p ./{output_folder}\n'
run_cuda_format_string = 'sbatch -o ./{output_folder} -p dcs -N {N} --ntasks-per-node={NR} --gres=gpu:{G} -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch={SY} --scheduling_algorithm={S} --num_aggregators={NA} --num_selectors={NS} --num_clients_per_selector={NC} --num_tasks={NT} --stats-path=./{output_folder} --end=600000 --mean_dur=120000 --prop_start=.05\n'
run_cpu_format_string = 'sbatch -o ./{output_folder} -p dcs -N {N} --ntasks-per-node={NR} --gres=gpu:{G} -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch={SY} --scheduling_algorithm={S} --num_aggregators={NA} --num_selectors={NS} --num_clients_per_selector={NC} --num_tasks={NT} --task_size={TS} --stats-path=./{output_folder} --end=600000 --mean_dur=120000 --prop_start=.05\n'

# SIM Strong Scaling
num_nodes = [1, 1, 1, 2, 2, 2, 2]
num_ranks_per_node = [1, 2, 4, 4, 8, 16, 32]
# 1, 2, 4, 8, 16, 32, 64
output_folder_template =  'Sim_Strong_Scaling/{num_ranks}Ranks/'
gpus = [1, 2, 4, 6, 6, 6, 6]
synch = [1, 2, 2, 2, 2, 2, 2]
S = 2
NA = 10
NS = 25
NC = 40
NT = 100

for i in range(len(num_nodes)):
    N = num_nodes[i]
    NR = num_ranks_per_node[i]
    G = gpus[i]
    SY = synch[i]
    output_folder = output_folder_template.format(num_ranks = N*NR)
    mkdir_cmd = mkdir_format_string.format(output_folder=output_folder)
    run_cmd = run_cuda_format_string.format(output_folder=output_folder,N=N,NR=NR,G=G,SY=SY,S=S,NA=NA,NS=NS,NC=NC,NT=NT)

    f.write(mkdir_cmd)
    f.write(run_cmd)
f.write('\n')

# SIM Weak Scaling
num_nodes = [1, 1, 1, 2, 2, 2, 2]
num_ranks_per_node = [1, 2, 4, 4, 8, 16, 32]
# 1, 2, 4, 8, 16, 32, 64
output_folder_template =  'Sim_Weak_Scaling/{num_ranks}Ranks/'
gpus = [1, 2, 4, 6, 6, 6, 6]
synch = [1, 2, 2, 2, 2, 2, 2]
S = 2
NA = 10
num_selectors = [1,2,4,8,16,32,64]
NC = 1000
NT = 100

for i in range(len(num_nodes)):
    N = num_nodes[i]
    NR = num_ranks_per_node[i]
    G = gpus[i]
    SY = synch[i]
    NS = num_selectors
    output_folder = output_folder_template.format(num_ranks = N*NR)
    mkdir_cmd = mkdir_format_string.format(output_folder=output_folder)
    run_cmd = run_cuda_format_string.format(output_folder=output_folder,N=N,NR=NR,G=G,SY=SY,S=S,NA=NA,NS=NS,NC=NC,NT=NT)

    f.write(mkdir_cmd)
    f.write(run_cmd)
f.write('\n')

# Phones Strong scaling
N = 2
NR = 32
# 1, 2, 4, 8, 16, 32, 64
output_folder_template =  'PhoneStrong_ScalingRCTA/{num_clients}Clients/'
G = 6
SY = 2
S = 2
num_aggregators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 100, 1000]
num_selectors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 10, 100]
num_clients_per_selector = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 750, 1000, 1000, 1000]
NT = 100
TS = 100

for i in range(len(num_aggregators)):
    NA = num_aggregators[i]
    NS = num_selectors[i]
    NC = num_clients_per_selector[i]
    output_folder = output_folder_template.format(num_clients = NC*NS)
    mkdir_cmd = mkdir_format_string.format(output_folder=output_folder)
    run_cmd = run_cpu_format_string.format(output_folder=output_folder,N=N,NR=NR,G=G,SY=SY,S=S,NA=NA,NS=NS,NC=NC,NT=NT,TS=TS)

    f.write(mkdir_cmd)
    f.write(run_cmd)
f.write('\n')

# Phones Weak scaling
N = 2
NR = 32
# 1, 2, 4, 8, 16, 32, 64
output_folder_template =  'PhoneWeak_ScalingRCTA/{num_clients}Clients_{num_subtasks}Subtasks/'
G = 6
SY = 2
S = 2
NT = 10
num_aggregators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
num_selectors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
num_clients_per_selector = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 750, 1000]
task_sizes = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 100, 1500, 2000]

for i in range(len(num_aggregators)):
    NA = num_aggregators[i]
    NS = num_selectors[i]
    NC = num_clients_per_selector[i]
    TS = task_sizes[i]
    output_folder = output_folder_template.format(num_clients = NC*NS, num_subtasks = TS)
    mkdir_cmd = mkdir_format_string.format(output_folder=output_folder)
    run_cmd = run_cpu_format_string.format(output_folder=output_folder,N=N,NR=NR,G=G,SY=SY,S=S,NA=NA,NS=NS,NC=NC,NT=NT,TS=TS)

    f.write(mkdir_cmd)
    f.write(run_cmd)
f.write('\n')

# Phones Strong scaling naive
N = 2
NR = 32
# 1, 2, 4, 8, 16, 32, 64
output_folder_template =  'PhoneStrong_ScalingNaive/{num_clients}Clients/'
G = 6
SY = 2
S = 1
num_aggregators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 100, 1000]
num_selectors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 10, 100]
num_clients_per_selector = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 750, 1000, 1000, 1000]
NT = 100
TS = 100

for i in range(len(num_aggregators)):
    NA = num_aggregators[i]
    NS = num_selectors[i]
    NC = num_clients_per_selector[i]
    output_folder = output_folder_template.format(num_clients = NC*NS)
    mkdir_cmd = mkdir_format_string.format(output_folder=output_folder)
    run_cmd = run_cpu_format_string.format(output_folder=output_folder,N=N,NR=NR,G=G,SY=SY,S=S,NA=NA,NS=NS,NC=NC,NT=NT,TS=TS)

    f.write(mkdir_cmd)
    f.write(run_cmd)
f.write('\n')

# Phones Weak scaling naive
N = 2
NR = 32
# 1, 2, 4, 8, 16, 32, 64
output_folder_template =  'PhoneWeak_ScalingNaive/{num_clients}Clients_{num_subtasks}Subtasks/'
G = 6
SY = 2
S = 1
NT = 10
num_aggregators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
num_selectors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
num_clients_per_selector = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 750, 1000]
task_sizes = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 100, 1500, 2000]

for i in range(len(num_aggregators)):
    NA = num_aggregators[i]
    NS = num_selectors[i]
    NC = num_clients_per_selector[i]
    TS = task_sizes[i]
    output_folder = output_folder_template.format(num_clients = NC*NS, num_subtasks = TS)
    mkdir_cmd = mkdir_format_string.format(output_folder=output_folder)
    run_cmd = run_cpu_format_string.format(output_folder=output_folder,N=N,NR=NR,G=G,SY=SY,S=S,NA=NA,NS=NS,NC=NC,NT=NT,TS=TS)

    f.write(mkdir_cmd)
    f.write(run_cmd)
f.write('\n')
