mkdir -p sample_out/

if [ "$1" = "1" ]
then
    # One node
    echo 'running one node sample'
    sbatch -o ./sample_out/output.%a.out -p dcs -N 1 --ntasks-per-node=4 --gres=gpu:4 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=4 --num_selectors=5 --num_clients_per_selector=10 --num_tasks=10 --stats-path=./sample_out --end=600000 --mean_dur=120000 --prop_start=.05
else
    # two nodes
    echo 'running two nodes sample'
    sbatch -o ./sample_out/output.%a.out -p dcs -N 2 --ntasks-per-node=4 --gres=gpu:4 -t 30 ./slurmSpectrumCUDA.sh --event-trace=1 --synch=2 --scheduling_algorithm=1 --num_aggregators=4 --num_selectors=5 --num_clients_per_selector=10 --num_tasks=10 --stats-path=./sample_out --end=600000 --mean_dur=120000 --prop_start=.05
fi