#!/bin/sh
#SBATCH -A NAISS2023-5-119 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 2-00:00:00

python src/baseline_sbert.py --train_path='TRoTR/datasets/pair-by-line/train.binary.jsonl' --dev_path='TRoTR/datasets/pair-by-line/dev.oov.ranking.jsonl' --loss 'contrastive' --model_type "siamese" --pretrained_model cardiffnlp/twitter-xlm-roberta-large-2022 --evaluation "correlation" --batch_size 4 --output_path CICCIO
