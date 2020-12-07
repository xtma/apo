count=0
version="1206"

algo="appo"
lamda=0.95
for gamma in 0.9 0.95 0.99 0.999
do
    mkdir -p "log/${algo}_g-${gamma}_l-${lamda}_${version}"
    for exp in Swimmer-v3 Ant-v3 HalfCheetah-v3
    do
        for seed in 1 2 3 4 5
            do
                python main.py \
                    --algo ${algo} \
                    --env_id ${exp} \
                    --run_ID ${seed} \
                    --cuda_idx ${count}  \
                    --gamma ${gamma}  \
                    --lamda ${lamda}  \
                    --rm_vb_coef 0 \
                    >> "log/${algo}_g-${gamma}_l-${lamda}_${version}/${exp}__s-${seed}"  2>&1 &
                count=$((($count+1) % 8))
            done
    done
done

gamma=0.99
for lamda in 0.8 0.9 0.99
do
    mkdir -p "log/${algo}_g-${gamma}_l-${lamda}_${version}"
    for exp in Swimmer-v3 Ant-v3 HalfCheetah-v3
    do
        for seed in 1 2 3 4 5
            do
                python main.py \
                    --algo ${algo} \
                    --env_id ${exp} \
                    --run_ID ${seed} \
                    --cuda_idx ${count}  \
                    --gamma ${gamma}  \
                    --lamda ${lamda}  \
                    --rm_vb_coef  \
                    >> "log/${algo}_g-${gamma}_l-${lamda}_${version}/${exp}__s-${seed}"  2>&1 &
                count=$((($count+1) % 8))
            done
    done
done

# for algo in "appo" "ac" "atrpo" "trpo" "ppo"
# do
#     mkdir -p "log/${algo}-${version}"
#     for exp in Swimmer-v3 Ant-v3 HalfCheetah-v3
#     do
#         for seed in 1 2 3 4 5
#             do
#                 python main.py \
#                     --algo ${algo} \
#                     --env_id ${exp} \
#                     --run_ID ${seed} \
#                     --cuda_idx ${count}  \
#                     >> "log/${algo}-${version}/${algo}-${exp}--s-${seed}"  2>&1 &
#                 count=$((($count+1) % 7))
#             done
#     done
# done

