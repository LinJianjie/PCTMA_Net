#!/usr/bin/env bash
python ../pointComNet/train/train_PCTMA_Net.py --batch_size 32 --epochs 300 --checkpoint_name  best_model_point_ppd_pct_gt_cmlp.pth  --best_name  best_model_point_ppd_pct_gt_cmlp.pth --load_checkPoints False --train True --evaluate False
