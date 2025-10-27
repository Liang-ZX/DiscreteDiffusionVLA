# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path "/path/to/xxx_chkpt" \
  --data_root_dir "/path/to/modified_libero_rlds" \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir "/path/to/checkpoints/ddopenvla-libero-spatial-from-ckpt" \
  --use_discrete_diffusion True \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 320005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity <WANDB_ENTITY> \
  --wandb_project <WANDB_PROJECT> \
  --resume True \
  --resume_step <resume_step> \
  --run_id_note "parallel_dec--8_acts_chunk--bin_acts--discrete_diffusion--3rd_person_img--wrist_img--proprio_state--$(date +%Y%m%d_%H%M)"
  