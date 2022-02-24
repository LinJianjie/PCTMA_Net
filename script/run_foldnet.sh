#!/usr/bin/env bash
python ../pointComNet/foldingnet/train_foldingNet.py  --batch_size 32 --epochs 300 --checkpoint_name  best_model_foldnet.pth  --best_name  best_model_foldnet.pth --load_checkPoints False --train True --evaluate True
