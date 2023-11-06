#!/bin/sh
#SBATCH -A NAISS2023-5-119 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 2-00:00:00

batch_size="8"
n_epochs="100"

echo "In-of-Vocabulary (IoV)"
python src/TRiC-model.py --batch_size "${batch_size}" --train_path TRoTR/datasets/pair-by-line/train.ranking.jsonl --dev_path TRoTR/datasets/pair-by-line/dev.ranking.jsonl --test_path TRoTR/datasets/pair-by-line/test.iov.ranking.jsonl --do_training --do_validation --do_prediction --stats_path stats/in-of-vocabulary --best_model_path TRoBERTaReg --n_epochs "${n_epochs}" --accum_iter 1

echo "Out-of-Vocabulary (OoV)"
python src/TRiC-model.py --batch_size "${batch_size}" --train_path TRoTR/datasets/pair-by-line/train.ranking.jsonl --dev_path TRoTR/datasets/pair-by-line/dev.ranking.jsonl --test_path TRoTR/datasets/pair-by-line/test.oov.ranking.jsonl --do_prediction --stats_path stats/out-of-vocabulary --best_model_path TRoBERTaReg
