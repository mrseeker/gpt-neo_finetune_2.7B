export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true
apt update
apt upgrade -y
apt install -y nano git python3.8 python3-numpy python3-pip cmake python3-arrow wget build-essential
pip3 install cython
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install pyarrow==0.17.1
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
TORCH_CUDA_ARCH_LIST="11.1" pip3 install .
cd
pip3 install transformers
pip3 install -r requirements.txt
pip3 install datasets==1.5.0 pyarrow==0.17.1 packaging
ds_report > /data/report.txt

#CODE HERE
deepspeed --num_gpus=4 run_clm.py \
--deepspeed ds_config_gptneo.json \
--model_name_or_path EleutherAI/gpt-neo-2.7B \
--model_type gpt_neo \
--do_train \
--fp16 \
--overwrite_cache \
--overwrite_output_dir \
--output_dir /data/finetuned \
--validation_file validation.csv \
--num_train_epochs 1 \
--max_train_samples 99200 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 2 \
--use_fast_tokenizer False \
--learning_rate 5e-06 \
--save_total_limit 1 \
--save_steps 400 \
--save_strategy steps \
--block_size 2048 \
--seed 5 \
--warmup_steps 10 \
--train_file fb-2048.map