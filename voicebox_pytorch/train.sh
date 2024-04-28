export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

accelerate launch \
    --mixed_precision=fp16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11135 \
    --num_processes 8 \
    --multi_gpu \
    train.py \
    --resume_training