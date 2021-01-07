count=0
version="1206"

# APPO, Swimmer
algo="appo"
exp="Swimmer-v3"
lamda=0.99
gamma=0.999
lr_eta=0.03
rm_vb_coef=0.3
log_dir="log/${algo}_g-${gamma}_l-${lamda}_e-${lr_eta}_v-${rm_vb_coef}_${version}"
mkdir -p ${log_dir}
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

# APPO, Ant
algo="appo"
exp="Ant-v3"
lamda=0.95
gamma=0.99
lr_eta=0.3
rm_vb_coef=0.3
log_dir="log/${algo}_g-${gamma}_l-${lamda}_e-${lr_eta}_v-${rm_vb_coef}_${version}"
mkdir -p ${log_dir}
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

# APPO, HalfCheetah
algo="appo"
exp="HalfCheetah-v3"
lamda=0.9
gamma=0.999
lr_eta=0.1
rm_vb_coef=1
log_dir="log/${algo}_g-${gamma}_l-${lamda}_e-${lr_eta}_v-${rm_vb_coef}_${version}"
mkdir -p ${log_dir}
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
    
# PPO, Swimmer
algo="ppo"
exp="Swimmer-v3"
lamda=0.99
gamma=0.999
log_dir="log/${algo}_g-${gamma}_l-${lamda}_${version}"
mkdir -p ${log_dir}
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
            >> "${log_dir}/${exp}__s-${seed}"  2>&1 &
        count=$((($count+1) % 8))
    done

# PPO, Ant
algo="ppo"
exp="Ant-v3"
lamda=0.8
gamma=0.99
log_dir="log/${algo}_g-${gamma}_l-${lamda}_${version}"
mkdir -p ${log_dir}
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
            >> "${log_dir}/${exp}__s-${seed}"  2>&1 &
        count=$((($count+1) % 8))
    done

# PPO, HalfCheetah
algo="ppo"
exp="HalfCheetah-v3"
lamda=0.9
gamma=0.9
log_dir="log/${algo}_g-${gamma}_l-${lamda}_${version}"
mkdir -p ${log_dir}
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
            >> "${log_dir}/${exp}__s-${seed}"  2>&1 &
        count=$((($count+1) % 8))
    done


