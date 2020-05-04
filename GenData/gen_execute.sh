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

mkir -p ./PhoneStrong_Scaling/10Clients/
sbatch -o ./PhoneStrong_Scaling/10Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=1 --num_selectors=1 --num_clients_per_selector=10 --num_tasks=100 --stats-path=./PhoneStrong_Scaling/10Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_Scaling/100Clients/
sbatch -o ./PhoneStrong_Scaling/100Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=2 --num_selectors=1 --num_clients_per_selector=100 --num_tasks=100 --stats-path=./PhoneStrong_Scaling/100Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_Scaling/200Clients/
sbatch -o ./PhoneStrong_Scaling/200Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=3 --num_selectors=1 --num_clients_per_selector=200 --num_tasks=100 --stats-path=./PhoneStrong_Scaling/200Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_Scaling/300Clients/
sbatch -o ./PhoneStrong_Scaling/300Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=4 --num_selectors=1 --num_clients_per_selector=300 --num_tasks=100 --stats-path=./PhoneStrong_Scaling/300Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_Scaling/400Clients/
sbatch -o ./PhoneStrong_Scaling/400Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=5 --num_selectors=1 --num_clients_per_selector=400 --num_tasks=100 --stats-path=./PhoneStrong_Scaling/400Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_Scaling/500Clients/
sbatch -o ./PhoneStrong_Scaling/500Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=6 --num_selectors=1 --num_clients_per_selector=500 --num_tasks=100 --stats-path=./PhoneStrong_Scaling/500Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneStrong_Scaling/600Clients/
sbatch -o ./PhoneStrong_Scaling/600Clients/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=7 --num_selectors=1 --num_clients_per_selector=600 --num_tasks=100 --stats-path=./PhoneStrong_Scaling/600Clients/ --end=600000 --mean_dur = 120000 --prop_start=.05

mkir -p ./PhoneWeak_Scaling/10Clients_10Tasks/
sbatch -o ./PhoneWeak_Scaling/10Clients_10Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=1 --num_selectors=1 --num_clients_per_selector=10 --num_tasks=10 --stats-path=./PhoneWeak_Scaling/10Clients_10Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_Scaling/100Clients_100Tasks/
sbatch -o ./PhoneWeak_Scaling/100Clients_100Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=2 --num_selectors=1 --num_clients_per_selector=100 --num_tasks=100 --stats-path=./PhoneWeak_Scaling/100Clients_100Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_Scaling/200Clients_200Tasks/
sbatch -o ./PhoneWeak_Scaling/200Clients_200Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=3 --num_selectors=1 --num_clients_per_selector=200 --num_tasks=200 --stats-path=./PhoneWeak_Scaling/200Clients_200Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_Scaling/300Clients_300Tasks/
sbatch -o ./PhoneWeak_Scaling/300Clients_300Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=4 --num_selectors=1 --num_clients_per_selector=300 --num_tasks=300 --stats-path=./PhoneWeak_Scaling/300Clients_300Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_Scaling/400Clients_400Tasks/
sbatch -o ./PhoneWeak_Scaling/400Clients_400Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=5 --num_selectors=1 --num_clients_per_selector=400 --num_tasks=400 --stats-path=./PhoneWeak_Scaling/400Clients_400Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_Scaling/500Clients_500Tasks/
sbatch -o ./PhoneWeak_Scaling/500Clients_500Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=6 --num_selectors=1 --num_clients_per_selector=500 --num_tasks=500 --stats-path=./PhoneWeak_Scaling/500Clients_500Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05
mkir -p ./PhoneWeak_Scaling/600Clients_600Tasks/
sbatch -o ./PhoneWeak_Scaling/600Clients_600Tasks/ -p dcs -N 2 --ntasks-per-node=32 --gres=gpu:6 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=2 --num_aggregators=7 --num_selectors=1 --num_clients_per_selector=600 --num_tasks=600 --stats-path=./PhoneWeak_Scaling/600Clients_600Tasks/ --end=600000 --mean_dur = 120000 --prop_start=.05

