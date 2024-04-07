ACTOR_MODEL_PATH=/mnt/bn/wuxibin-hl-dev/cache/models--codellama--CodeLlama-34b-hf/snapshots/d3e967887d285343b8e239e26c6778c26931a536/
CRITIC_MODEL_PATH=/mnt/bn/wuxibin-hl-dev/cache/models--codellama--CodeLlama-34b-hf/snapshots/d3e967887d285343b8e239e26c6778c26931a536/
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
OUTPUT=
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step3_llama
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

ray job submit --address="http://[fdbd:dc02:19:792::203]:10435" \
    --runtime-env=training/step3_rlhf_finetuning/training_scripts/llama2/runtime.yaml \
    --no-wait \
    -- python3 training/step3_rlhf_finetuning/main.py \
    --enable_ray \
    --num_nodes 1 \
    --num_gpus_per_node 8 \
    --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons \
    --data_split 2,4,4 \
    --actor_model_name_or_path $ACTOR_MODEL_PATH \
    --critic_model_name_or_path $CRITIC_MODEL_PATH \
    --num_padding_at_beginning 1 \
    --per_device_generation_batch_size 8 \
    --per_device_training_batch_size 8 \
    --generation_batches 16 \
    --gradient_accumulation_steps 2 \
    --ppo_epochs 1 \
    --max_answer_seq_len 1024 \
    --max_prompt_seq_len 1024 \
    --actor_learning_rate ${Actor_Lr} \
    --critic_learning_rate ${Critic_Lr} \
    --actor_weight_decay 0.1 \
    --critic_weight_decay 0.1 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --actor_gradient_checkpointing \
    --critic_gradient_checkpointing \
    --actor_dropout 0.0 \
    --num_warmup_steps 100 \
    --deepspeed --seed 1234 \
    --actor_zero_stage $ACTOR_ZERO_STAGE \
    --critic_zero_stage $CRITIC_ZERO_STAGE \
    --enable_hybrid_engine \
    --inference_tp_size 1 \
    --release_inference_cache \
    --offload \
    --offload_reference_model \
    --output_dir $OUTPUT