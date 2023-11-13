#!/bin/sh
#SBATCH -A NAISS2023-5-119 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 2-00:00:00

batch_size=32

declare -a models=("bert-base-uncased" "bert-base-multilingual-uncased" "bert-large-uncased")
for model in "${models[@]}"
do
   echo "TRiC-BERT - ${model}"
   #python src/TRiC-BERT.py --model "${model}" --batch_size "${batch_size}" --device cuda
   echo "TRiC-BERT-nsp - ${model}"
   #python src/TRiC-BERT-nsp.py --model "${model}" --batch_size "${batch_size}" --device cuda
done

declare -a models=("all-mpnet-base-v2" "multi-qa-mpnet-base-dot-v1" "all-distilroberta-v1" "all-MiniLM-L12-v2" "multi-qa-distilbert-cos-v1" "all-MiniLM-L6-v2" "multi-qa-MiniLM-L6-cos-v1" "paraphrase-multilingual-mpnet-base-v2" "paraphrase-albert-small-v2" "paraphrase-multilingual-MiniLM-L12-v2" "paraphrase-MiniLM-L3-v2" "distiluse-base-multilingual-cased-v1" "distiluse-base-multilingual-cased-v2" "multi-qa-MiniLM-L6-dot-v1" "multi-qa-distilbert-dot-v1" "multi-qa-mpnet-base-dot-v1" "multi-qa-MiniLM-L6-cos-v1" "multi-qa-distilbert-cos-v1" "multi-qa-mpnet-base-cos-v1" "msmarco-distilbert-base-tas-b" "msmarco-distilbert-dot-v5" "msmarco-bert-base-dot-v5" "msmarco-MiniLM-L6-cos-v5" "msmarco-MiniLM-L12-cos-v5" "msmarco-distilbert-cos-v5")
for model in "${models[@]}"
do
   echo "TRiC-sBERT (BiEncoder) - ${model}"
   #python src/TRiC-sBERT-BiEncoder.py --model "${model}" --batch_size "${batch_size}" --device cuda
done

declare -a models=("cross-encoder/ms-marco-TinyBERT-L-2-v2" "cross-encoder/ms-marco-MiniLM-L-2-v2" "cross-encoder/ms-marco-MiniLM-L-4-v2" "cross-encoder/ms-marco-MiniLM-L-6-v2" "cross-encoder/ms-marco-MiniLM-L-12-v2" "cross-encoder/qnli-distilroberta-base" "cross-encoder/qnli-electra-base" "cross-encoder/stsb-TinyBERT-L-4" "cross-encoder/stsb-distilroberta-base" "cross-encoder/stsb-roberta-base" "cross-encoder/stsb-roberta-large" "cross-encoder/quora-distilroberta-base" "cross-encoder/quora-roberta-base" "cross-encoder/quora-roberta-large" "cross-encoder/nli-deberta-v3-base" "cross-encoder/nli-deberta-base" "cross-encoder/nli-deberta-v3-xsmall" "cross-encoder/nli-deberta-v3-small" "cross-encoder/nli-roberta-base" "cross-encoder/nli-MiniLM2-L6-H768" "cross-encoder/nli-distilroberta-base")
for model in "${models[@]}"
do
   echo "TRiC-sBERT (CrossEncoder) - ${model}"
   python src/TRiC-sBERT-CrossEncoder.py --model "${model}" --batch_size "${batch_size}" --device cuda
done
