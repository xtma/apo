count=0
version="0803"

# appo
algo="appo"
mkdir -p "log/${algo}-${version}"
for exp in Swimmer-v3 Ant-v3 HalfCheetah-v3 # Pendulum-v0 
do
    for seed in 1 2 3 4 5
        do
            python appo.py --env_id ${exp} --run_ID ${seed} --cuda_idx ${count}  >> "log/${algo}-${version}/${algo}-${exp}--s-${seed}"  2>&1 &
            count=$((($count+1) % 7))
        done
done

# ppo
algo="ppo"
mkdir -p "log/${algo}-${version}"
for exp in Swimmer-v3 Ant-v3 HalfCheetah-v3 # Pendulum-v0 
do
    for seed in 1 2 3 4 5
        do
            python ppo.py --env_id ${exp} --run_ID ${seed} --cuda_idx ${count}  >> "log/${algo}-${version}/${algo}-${exp}--s-${seed}"  2>&1 &
            count=$((($count+1) % 7))
        done
done

# ppo_norm
algo="ppo_norm"
mkdir -p "log/${algo}-${version}"
for exp in Swimmer-v3 Ant-v3 HalfCheetah-v3 # Pendulum-v0 
do
    for seed in 1 2 3 4 5
        do
            python ppo_norm.py --env_id ${exp} --run_ID ${seed} --cuda_idx ${count}  >> "log/${algo}-${version}/${algo}-${exp}--s-${seed}"  2>&1 &
            count=$((($count+1) % 7))
        done
done