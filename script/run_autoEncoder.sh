#!/usr/bin/env bash
python ../pointComNet/train/train_point_ae.py --batch_size 32 --epochs 150 --checkpoint_name  best_model_point_ae_pt.pth  --best_name  best_model_point_ae_pt.pth --load_checkPoints False --train True --evaluate False
