#!/bin/bash

# export TRANSFORMERS_OFFLINE=1
cuda=0
dataset=cnn_dailymail #paintings, art, anime, DB, cnn_dailymail
# seed=14 #paintings, art, anime, DB, cnn_dailymail
white_model=llama2-7b #vicuna-13b vicuna-7b llama2-7b promtist sft
black_model=claude3.5 # sd1.5 dreamlike sdXL gpt-image-1 gpt3.5-turbo gpt4-turbo claude3.5
optimizer=spsa #mmt spsa 
h=0.001
lr=1e-4 #1e-5 2e-5 5e-5 1e-4 1e-1 1e-2
batch_size=2   #1 --> single, other --> batch
epochs=2
train_size=64 #16 32 64 128 256
test_size=128 #16 32 64 128 256
n_directions=5
lora_rank=4 #4 8
metric=total #"aesthetic, clip, pick, total"
#image_evaluator
lambda_1=0.2
lambda_2=5
lambda_3=0.05
if [ "$black_model" = "gpt3.5-turbo" ] || [ "$black_model" = "gpt4-turbo" ] || [ "$black_model" = "claude3.5" ]; then
    lambda_1=1
    lambda_2=0.03
    lambda_3=1
fi
gene_image=False #True False
soft_train=True #True False
soft_epochs=2
soft_lr=0.1
mu=0.1
intrinsic_dim=10
n_prompt_tokens=5
soft_train_batches=16
soft_n_directions=1
random_proj=uniform # normal uniform
debug=False  # True False
for seed in 14 42 81;do #14 42 81
# for dataset in cnn_dailymail;do #14 42 81
    # cmd="python -m debugpy --listen 5678 --wait-for-client test.py --seed $seed"
    # 构建基础目录路径（包含主要参数）
    base_dir="./result/${dataset}/${white_model}_${black_model}_opt_${optimizer}_proj${random_proj}/samples_${train_size}_${test_size}_batch${batch_size}_lora_rank${lora_rank}/ep${epochs}_dir${n_directions}_lr${lr}_h${h}"
    
    # 根据soft_train参数决定是否添加软提示相关参数到路径
    if [ "$soft_train" = "True" ]; then
        soft_params_dir="/soft_ep${soft_epochs}_lr${soft_lr}_mu${mu}/dim${intrinsic_dim}_tokens${n_prompt_tokens}_batches${soft_train_batches}_soft_n_dir${soft_n_directions}"
    fi
    
    output_path="${base_dir}/seed${seed}/output.txt"
    output_dir=$(dirname "$output_path")
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    fi

    if [ "$soft_train" = "True" ]; then
        soft_output_path="${output_dir}/${soft_params_dir}/output.txt"
    fi
    soft_output_dir=$(dirname "$soft_output_path")
    if [ ! -d "$soft_output_dir" ]; then
        mkdir -p "$soft_output_dir"
    fi



    if [ "$debug" = "True" ]; then
        cmd="python -m debugpy --listen 5679 --wait-for-client main.py"
    else
        cmd="python3 main.py"
    fi
    cmd="$cmd --cuda $cuda\
    --dataset $dataset\
    --train_size $train_size\
    --test_size $test_size\
    --white_model $white_model\
    --black_model $black_model\
    --optimizer $optimizer\
    --h $h\
    --lr $lr\
    --metric $metric\
    --batch_size $batch_size\
    --epochs $epochs\
    --n_directions $n_directions\
    --lambda_1 $lambda_1\
    --lambda_2 $lambda_2\
    --lambda_3 $lambda_3\
    --output_dir $output_dir\
    --soft_output_dir $soft_output_dir\
    --gene_image $gene_image\
    --lora_rank $lora_rank\
    --debug $debug\
    --soft_train_batches $soft_train_batches\
    --soft_train $soft_train\
    --soft_epochs $soft_epochs\
    --soft_lr $soft_lr\
    --mu $mu\
    --n_prompt_tokens $n_prompt_tokens\
    --random_proj $random_proj\
    --intrinsic_dim $intrinsic_dim\
    --soft_n_directions $soft_n_directions\
    --seed $seed"
    # --seed $seed > $soft_output_path 2>&1"
    echo $cmd
    eval $cmd
    echo "$cmd" >> "$soft_output_path"
done
# -m debugpy --listen 5679 --wait-for-client 