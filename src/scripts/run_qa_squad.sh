# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
#     run_question_answering.py \
#     --model_name_or_path t5-large \
#     --do_train \
#     --do_eval \
#     --dataset_name squad \
#     --context_column context \
#     --question_column question \
#     --answer_column answers \
#     --output_dir ./save/squad_t5_large/ \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 32 \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --save_steps 5475 \
#     --learning_rate 1e-4 \
#     --num_train_epochs 10 \
#     --max_seq_length 512 \

    # FREE
    # --output_hidden_states_decoder True \
    # --intermediate_loss_fn shallowdeep_kd_dyna \
    # --shallow_exit_layer 6 \
    # --distill_layer_alpha 0.5 \
    # --do_layer_transformation False \

    # CALM
    # --output_hidden_states_decoder True \
    # --intermediate_loss_fn weighted_ce \

CUDA_VISIBLE_DEVICES=0 python -m run_question_answering \
    --model_name_or_path google-t5/t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5-base/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize False \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_seq_length 512 \
    --use_early_exit True \
    --exit_conf_type JSD_contrastive_confidence \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 7 \
    --include_inputs_for_metrics True \
    --max_eval_samples 10 \
    --use_auth_token True \
    --count_flops False \




   # JSD_contrastive_confidence

    # FREE
    # --use_shallow_deep True \
    # --shallow_exit_layer 6 \
    # --shallow2deep_conf_type softmax \
    # --shallow2deep_conf_threshold 0.9 \
    # --use_adap_threshold True \ # to use adaptive threshold

    #CALM 
    

    # static-exiting
    # --static_exit_layer 6 \

    # evaluate only performance
    # --deploy_scenario False \
    # --per_device_eval_batch_size 8 \

