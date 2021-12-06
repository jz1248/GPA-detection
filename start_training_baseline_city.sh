GPU_ID=0,1,2,3
CUDA_VISIBLE_DEVICES=$GPU_ID python train_baseline.py --dataset city_multi \
                    --model_config baseline --net res50 --bs 3 --nw 1 \
                    --epochs 10 --lr 0.001 --lr_decay_step 6 \
                    --lr_decay_gamma 0.1 --cuda --use_tfb
