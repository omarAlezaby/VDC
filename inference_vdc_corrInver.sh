python scripts/CondMapping.py \
--inference-imgs "datasets/rain/source" \
--outdir "outputs/vdc_inference_corrInver_rain" \
--ddim_steps 100 \
--scale 7.0 \
--strength 0.1 \
--opt_inversion true \
--opt_inversion_itrs 20 \
--cond_path "vdc_train_rain/learned_conds.npy"
