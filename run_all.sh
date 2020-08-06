count=0
version="0806"

for algo in "appo" "ac" "atrpo" "trpo" "ppo"
do
    mkdir -p "log/${algo}-${version}"
    for exp in Swimmer-v3 Ant-v3 HalfCheetah-v3 # Pendulum-v0 
    do
        for seed in 1 2 3 4 5
            do
                python main.py \
                    --algo ${algo} \
                    --env_id ${exp} \
                    --run_ID ${seed} \
                    --cuda_idx ${count}  \
                    >> "log/${algo}-${version}/${algo}-${exp}--s-${seed}"  2>&1 &
                count=$((($count+1) % 7))
            done
    done
done

