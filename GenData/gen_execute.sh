mkir -p ./Sim_Strong_Scaling/1Ranks/
sbatch -o ./Sim_Strong_Scaling/1Ranks/ -p dcs -N 1 --ntasks-per-node=1 --gres=gpu:1 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=1 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=25 --num_clients_per_selector=40 --num_tasks=100 --stats-path=./Sim_Strong_Scaling/1Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Strong_Scaling/2Ranks/
sbatch -o ./Sim_Strong_Scaling/2Ranks/ -p dcs -N 1 --ntasks-per-node=2 --gres=gpu:2 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=25 --num_clients_per_selector=40 --num_tasks=100 --stats-path=./Sim_Strong_Scaling/2Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Strong_Scaling/4Ranks/
sbatch -o ./Sim_Strong_Scaling/4Ranks/ -p dcs -N 1 --ntasks-per-node=4 --gres=gpu:4 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=25 --num_clients_per_selector=40 --num_tasks=100 --stats-path=./Sim_Strong_Scaling/4Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Strong_Scaling/8Ranks/
sbatch -o ./Sim_Strong_Scaling/8Ranks/ -p dcs -N 2 --ntasks-per-node=4 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=25 --num_clients_per_selector=40 --num_tasks=100 --stats-path=./Sim_Strong_Scaling/8Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Strong_Scaling/16Ranks/
sbatch -o ./Sim_Strong_Scaling/16Ranks/ -p dcs -N 2 --ntasks-per-node=8 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=25 --num_clients_per_selector=40 --num_tasks=100 --stats-path=./Sim_Strong_Scaling/16Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Strong_Scaling/32Ranks/
sbatch -o ./Sim_Strong_Scaling/32Ranks/ -p dcs -N 2 --ntasks-per-node=16 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=25 --num_clients_per_selector=40 --num_tasks=100 --stats-path=./Sim_Strong_Scaling/32Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Strong_Scaling/64Ranks/
sbatch -o ./Sim_Strong_Scaling/64Ranks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=25 --num_clients_per_selector=40 --num_tasks=100 --stats-path=./Sim_Strong_Scaling/64Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05

mkir -p ./Sim_Weak_Scaling/1Ranks/
sbatch -o ./Sim_Weak_Scaling/1Ranks/ -p dcs -N 1 --ntasks-per-node=1 --gres=gpu:1 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=1 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=[1, 2, 4, 8, 16, 32, 64] --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./Sim_Weak_Scaling/1Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Weak_Scaling/2Ranks/
sbatch -o ./Sim_Weak_Scaling/2Ranks/ -p dcs -N 1 --ntasks-per-node=2 --gres=gpu:2 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=[1, 2, 4, 8, 16, 32, 64] --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./Sim_Weak_Scaling/2Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Weak_Scaling/4Ranks/
sbatch -o ./Sim_Weak_Scaling/4Ranks/ -p dcs -N 1 --ntasks-per-node=4 --gres=gpu:4 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=[1, 2, 4, 8, 16, 32, 64] --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./Sim_Weak_Scaling/4Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Weak_Scaling/8Ranks/
sbatch -o ./Sim_Weak_Scaling/8Ranks/ -p dcs -N 2 --ntasks-per-node=4 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=[1, 2, 4, 8, 16, 32, 64] --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./Sim_Weak_Scaling/8Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Weak_Scaling/16Ranks/
sbatch -o ./Sim_Weak_Scaling/16Ranks/ -p dcs -N 2 --ntasks-per-node=8 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=[1, 2, 4, 8, 16, 32, 64] --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./Sim_Weak_Scaling/16Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Weak_Scaling/32Ranks/
sbatch -o ./Sim_Weak_Scaling/32Ranks/ -p dcs -N 2 --ntasks-per-node=16 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=[1, 2, 4, 8, 16, 32, 64] --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./Sim_Weak_Scaling/32Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./Sim_Weak_Scaling/64Ranks/
sbatch -o ./Sim_Weak_Scaling/64Ranks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=[1, 2, 4, 8, 16, 32, 64] --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./Sim_Weak_Scaling/64Ranks/ --end=600000 --mean_dur = 120000 --prop_start=.05

mkir -p ./PhoneStrong_ScalingRCTA/10Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/10Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=1 --num_selectors=1 --num_clients_per_selector=10 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/10Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/100Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/100Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=2 --num_selectors=1 --num_clients_per_selector=100 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/100Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/200Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/200Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=3 --num_selectors=1 --num_clients_per_selector=200 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/200Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/300Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/300Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=4 --num_selectors=1 --num_clients_per_selector=300 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/300Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/400Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/400Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=5 --num_selectors=1 --num_clients_per_selector=400 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/400Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/500Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/500Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=6 --num_selectors=1 --num_clients_per_selector=500 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/500Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/600Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/600Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=7 --num_selectors=1 --num_clients_per_selector=600 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/600Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/700Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/700Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=8 --num_selectors=1 --num_clients_per_selector=700 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/700Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/800Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/800Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=9 --num_selectors=1 --num_clients_per_selector=800 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/800Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/900Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/900Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=1 --num_clients_per_selector=900 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/900Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/2000Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/2000Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=15 --num_selectors=2 --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/2000Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/1500Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/1500Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=20 --num_selectors=2 --num_clients_per_selector=750 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/1500Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/10000Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/10000Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=100 --num_selectors=10 --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/10000Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingRCTA/100000Clients/
sbatch -o ./PhoneStrong_ScalingRCTA/100000Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=1000 --num_selectors=100 --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./PhoneStrong_ScalingRCTA/100000Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05

mkir -p ./PhoneWeak_ScalingRCTA/10Clients_10Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/10Clients_10Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=1 --num_selectors=1 --num_clients_per_selector=10 --num_tasks=10 --stats-path=./PhoneWeak_ScalingRCTA/10Clients_10Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/100Clients_100Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/100Clients_100Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=2 --num_selectors=1 --num_clients_per_selector=100 --num_tasks=100 --stats-path=./PhoneWeak_ScalingRCTA/100Clients_100Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/200Clients_200Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/200Clients_200Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=3 --num_selectors=1 --num_clients_per_selector=200 --num_tasks=200 --stats-path=./PhoneWeak_ScalingRCTA/200Clients_200Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/300Clients_300Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/300Clients_300Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=4 --num_selectors=1 --num_clients_per_selector=300 --num_tasks=300 --stats-path=./PhoneWeak_ScalingRCTA/300Clients_300Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/400Clients_400Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/400Clients_400Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=5 --num_selectors=1 --num_clients_per_selector=400 --num_tasks=400 --stats-path=./PhoneWeak_ScalingRCTA/400Clients_400Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/500Clients_500Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/500Clients_500Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=6 --num_selectors=1 --num_clients_per_selector=500 --num_tasks=500 --stats-path=./PhoneWeak_ScalingRCTA/500Clients_500Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/600Clients_600Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/600Clients_600Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=7 --num_selectors=1 --num_clients_per_selector=600 --num_tasks=600 --stats-path=./PhoneWeak_ScalingRCTA/600Clients_600Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/700Clients_700Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/700Clients_700Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=8 --num_selectors=1 --num_clients_per_selector=700 --num_tasks=700 --stats-path=./PhoneWeak_ScalingRCTA/700Clients_700Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/800Clients_800Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/800Clients_800Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=9 --num_selectors=1 --num_clients_per_selector=800 --num_tasks=800 --stats-path=./PhoneWeak_ScalingRCTA/800Clients_800Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/900Clients_900Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/900Clients_900Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=10 --num_selectors=1 --num_clients_per_selector=900 --num_tasks=900 --stats-path=./PhoneWeak_ScalingRCTA/900Clients_900Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/2000Clients_100Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/2000Clients_100Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=15 --num_selectors=2 --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./PhoneWeak_ScalingRCTA/2000Clients_100Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingRCTA/1500Clients_1500Tasks/
sbatch -o ./PhoneWeak_ScalingRCTA/1500Clients_1500Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=20 --num_selectors=2 --num_clients_per_selector=750 --num_tasks=1500 --stats-path=./PhoneWeak_ScalingRCTA/1500Clients_1500Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05

mkir -p ./PhoneStrong_ScalingNaive/10Clients/
sbatch -o ./PhoneStrong_ScalingNaive/10Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=1 --num_selectors=1 --num_clients_per_selector=10 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/10Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/100Clients/
sbatch -o ./PhoneStrong_ScalingNaive/100Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=2 --num_selectors=1 --num_clients_per_selector=100 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/100Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/200Clients/
sbatch -o ./PhoneStrong_ScalingNaive/200Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=3 --num_selectors=1 --num_clients_per_selector=200 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/200Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/300Clients/
sbatch -o ./PhoneStrong_ScalingNaive/300Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=4 --num_selectors=1 --num_clients_per_selector=300 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/300Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/400Clients/
sbatch -o ./PhoneStrong_ScalingNaive/400Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=5 --num_selectors=1 --num_clients_per_selector=400 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/400Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/500Clients/
sbatch -o ./PhoneStrong_ScalingNaive/500Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=6 --num_selectors=1 --num_clients_per_selector=500 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/500Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/600Clients/
sbatch -o ./PhoneStrong_ScalingNaive/600Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=7 --num_selectors=1 --num_clients_per_selector=600 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/600Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/700Clients/
sbatch -o ./PhoneStrong_ScalingNaive/700Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=8 --num_selectors=1 --num_clients_per_selector=700 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/700Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/800Clients/
sbatch -o ./PhoneStrong_ScalingNaive/800Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=9 --num_selectors=1 --num_clients_per_selector=800 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/800Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/900Clients/
sbatch -o ./PhoneStrong_ScalingNaive/900Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=10 --num_selectors=1 --num_clients_per_selector=900 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/900Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/2000Clients/
sbatch -o ./PhoneStrong_ScalingNaive/2000Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=15 --num_selectors=2 --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/2000Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/1500Clients/
sbatch -o ./PhoneStrong_ScalingNaive/1500Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=20 --num_selectors=2 --num_clients_per_selector=750 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/1500Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/10000Clients/
sbatch -o ./PhoneStrong_ScalingNaive/10000Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=100 --num_selectors=10 --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/10000Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_ScalingNaive/100000Clients/
sbatch -o ./PhoneStrong_ScalingNaive/100000Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=1000 --num_selectors=100 --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./PhoneStrong_ScalingNaive/100000Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05

mkir -p ./PhoneWeak_ScalingNaive/10Clients_10Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/10Clients_10Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=1 --num_selectors=1 --num_clients_per_selector=10 --num_tasks=10 --stats-path=./PhoneWeak_ScalingNaive/10Clients_10Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/100Clients_100Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/100Clients_100Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=2 --num_selectors=1 --num_clients_per_selector=100 --num_tasks=100 --stats-path=./PhoneWeak_ScalingNaive/100Clients_100Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/200Clients_200Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/200Clients_200Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=3 --num_selectors=1 --num_clients_per_selector=200 --num_tasks=200 --stats-path=./PhoneWeak_ScalingNaive/200Clients_200Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/300Clients_300Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/300Clients_300Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=4 --num_selectors=1 --num_clients_per_selector=300 --num_tasks=300 --stats-path=./PhoneWeak_ScalingNaive/300Clients_300Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/400Clients_400Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/400Clients_400Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=5 --num_selectors=1 --num_clients_per_selector=400 --num_tasks=400 --stats-path=./PhoneWeak_ScalingNaive/400Clients_400Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/500Clients_500Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/500Clients_500Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=6 --num_selectors=1 --num_clients_per_selector=500 --num_tasks=500 --stats-path=./PhoneWeak_ScalingNaive/500Clients_500Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/600Clients_600Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/600Clients_600Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=7 --num_selectors=1 --num_clients_per_selector=600 --num_tasks=600 --stats-path=./PhoneWeak_ScalingNaive/600Clients_600Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/700Clients_700Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/700Clients_700Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=8 --num_selectors=1 --num_clients_per_selector=700 --num_tasks=700 --stats-path=./PhoneWeak_ScalingNaive/700Clients_700Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/800Clients_800Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/800Clients_800Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=9 --num_selectors=1 --num_clients_per_selector=800 --num_tasks=800 --stats-path=./PhoneWeak_ScalingNaive/800Clients_800Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/900Clients_900Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/900Clients_900Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=10 --num_selectors=1 --num_clients_per_selector=900 --num_tasks=900 --stats-path=./PhoneWeak_ScalingNaive/900Clients_900Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/2000Clients_100Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/2000Clients_100Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=15 --num_selectors=2 --num_clients_per_selector=1000 --num_tasks=100 --stats-path=./PhoneWeak_ScalingNaive/2000Clients_100Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_ScalingNaive/1500Clients_1500Tasks/
sbatch -o ./PhoneWeak_ScalingNaive/1500Clients_1500Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCPU.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=20 --num_selectors=2 --num_clients_per_selector=750 --num_tasks=1500 --stats-path=./PhoneWeak_ScalingNaive/1500Clients_1500Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05

