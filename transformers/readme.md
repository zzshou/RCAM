pip install torch==1.5.1 torchvision==0.6.1
pip install transformer

# train
CUDA_VISIBLE_DEVICES=1 python run_multiple_choice.py \
--task_name semEval \
--model_name_or_path roberta-base \
--data_dir "sem_data/" \
--output_dir "out3" \
--do_train \
--do_eval \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--cache_dir "test"\
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output

# eval
python run_multiple_choice.py \
--task_name semEval \
--model_name_or_path "out/" \
--data_dir "sem_data/" \
--output_dir "out2" \
--do_eval \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--gradient_accumulation_steps 2 

