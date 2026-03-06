python scripts/CondMapping.py \
--inference-imgs "datasets/rain/source" \
--outdir "outputs/vdc_inference_rain" \
--ddim_steps 100 \
--scale 7.0 \
--strength 0.1 \
--opt_inversion false \
--cond_path "vdc_train_rain/learned_conds.npy"
