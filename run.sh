#!/bin/sh
#SBATCH -A NAISS2023-5-119 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1 # We're launching 1 nodes with A100 GPU
#SBATCH -t 2-00:00:00

batch_size="8"
echo "In-of-Vocabulary (IoV)"
python src/TRiC-model.py --batch_size "${batch_size}" --train_path TRoTR/datasets/in-of-vocabulary/pair-by-line/train-ranking.jsonl --dev_path TRoTR/datasets/in-of-vocabulary/pair-by-line/dev-ranking.jsonl --test_path TRoTR/datasets/in-of-vocabulary/pair-by-line/test-binary.jsonl --do_validation --do_training --do_prediction --stats_path stats/in-of-vocabulary --best_model_path TRoBERTa_IoV --n_epochs 10

echo "Out-of-Vocabulary (OoV)"
python src/TRiC-model.py --batch_size "${batch_size}" --train_path TRoTR/datasets/out-of-vocabulary/pair-by-line/train-ranking.jsonl --dev_path TRoTR/datasets/out-of-vocabulary/pair-by-line/dev-ranking.jsonl --test_path TRoTR/datasets/out-of-vocabulary/pair-by-line/test-binary.jsonl --do_validation --do_training --do_prediction --stats_path stats/out-of-vocabulary --best_model_path TRoBERTa_OoV --n_epochs 10


