count=0
version="1206"

# algo="appo"
# lamda=0.95
# gamma=1
# lr_eta=0.1
# for rm_vb_coef in 0 0.03 0.1 0.3 1
# do
#     log_dir="log/${algo}_g-${gamma}_l-${lamda}_e-${lr_eta}_v-${rm_vb_coef}_${version}"
#     mkdir -p ${log_dir}
#     for exp in Swimmer-v3 Ant-v3 HalfCheetah-v3
#     do
#         for seed in 1 2 3 4 5
#             do
#                 python main.py \
#                     --algo ${algo} \
#                     --env_id ${exp} \
#                     --run_ID ${seed} \
#                     --cuda_idx ${count}  \
#                     --gamma ${gamma}  \
#                     --lamda ${lamda}  \
#                     --lr_eta ${lr_eta} \
#                     --rm_vb_coef ${rm_vb_coef} \
#                     >> "${log_dir}/${exp}__s-${seed}"  2>&1 &
#                 count=$((($count+1) % 8))
#             done
#     done
# done

algo="appo"
lamda=0.95
gamma=1
rm_vb_coef=0.1
for lr_eta in 0.3 0.03
do
    log_dir="log/${algo}_g-${gamma}_l-${lamda}_e-${lr_eta}_v-${rm_vb_coef}_${version}"
    mkdir -p ${log_dir}
    for exp in Hopper-v3 Walker2d-v3 Humanoid-v3 Swimmer-v3 Ant-v3 HalfCheetah-v3
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
                    --lr_eta ${lr_eta} \
                    --rm_vb_coef ${rm_vb_coef} \
                    >> "${log_dir}/${exp}__s-${seed}"  2>&1 &
                count=$((($count+1) % 8))
            done
    done
done

# algo="appo"
# lamda=0.95
# for gamma in 0.9 0.95 0.99 0.999
# do
#     mkdir -p "log/${algo}_g-${gamma}_l-${lamda}_${version}"
#     for exp in Hopper-v3 Walker2d-v3 Humanoid-v3 # Swimmer-v3 Ant-v3 HalfCheetah-v3
#     do
#         for seed in 1 2 3 4 5
#             do
#                 python main.py \
#                     --algo ${algo} \
#                     --env_id ${exp} \
#                     --run_ID ${seed} \
#                     --cuda_idx ${count}  \
#                     --gamma ${gamma}  \
#                     --lamda ${lamda}  \
#                     --rm_vb_coef 0 \
#                     >> "log/${algo}_g-${gamma}_l-${lamda}_${version}/${exp}__s-${seed}"  2>&1 &
#                 count=$((($count+1) % 8))
#             done
#     done
# done

# gamma=0.99
# for lamda in 0.8 0.9 0.99
# do
#     mkdir -p "log/${algo}_g-${gamma}_l-${lamda}_${version}"
#     for exp in Hopper-v3 Walker2d-v3 Humanoid-v3 # Swimmer-v3 Ant-v3 HalfCheetah-v3
#     do
#         for seed in 1 2 3 4 5
#             do
#                 python main.py \
#                     --algo ${algo} \
#                     --env_id ${exp} \
#                     --run_ID ${seed} \
#                     --cuda_idx ${count}  \
#                     --gamma ${gamma}  \
#                     --lamda ${lamda}  \
#                     --rm_vb_coef 0 \
#                     >> "log/${algo}_g-${gamma}_l-${lamda}_${version}/${exp}__s-${seed}"  2>&1 &
#                 count=$((($count+1) % 8))
#             done
#     done
# done

