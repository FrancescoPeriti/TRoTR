batch_size=32
declare -a k_folds=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
declare -a biencoder_models=("all-mpnet-base-v2" "multi-qa-mpnet-base-dot-v1" "all-distilroberta-v1" "all-MiniLM-L12-v2" "multi-qa-distilbert-cos-v1" "all-MiniLM-L6-v2" "multi-qa-MiniLM-L6-cos-v1" "paraphrase-multilingual-mpnet-base-v2" "paraphrase-albert-small-v2" "paraphrase-multilingual-MiniLM-L12-v2" "paraphrase-MiniLM-L3-v2" "distiluse-base-multilingual-cased-v1" "distiluse-base-multilingual-cased-v2" "multi-qa-MiniLM-L6-dot-v1" "multi-qa-distilbert-dot-v1" "multi-qa-mpnet-base-dot-v1" "multi-qa-MiniLM-L6-cos-v1" "multi-qa-distilbert-cos-v1" "multi-qa-mpnet-base-cos-v1" "msmarco-distilbert-base-tas-b" "msmarco-distilbert-dot-v5" "msmarco-bert-base-dot-v5" "msmarco-MiniLM-L6-cos-v5" "msmarco-MiniLM-L12-cos-v5" "msmarco-distilbert-cos-v5")
declare -a crossencoder_models=("cross-encoder/ms-marco-TinyBERT-L-2-v2" "cross-encoder/ms-marco-MiniLM-L-2-v2" "cross-encoder/ms-marco-MiniLM-L-4-v2" "cross-encoder/ms-marco-MiniLM-L-6-v2" "cross-encoder/ms-marco-MiniLM-L-12-v2" "cross-encoder/qnli-distilroberta-base" "cross-encoder/qnli-electra-base" "cross-encoder/stsb-TinyBERT-L-4" "cross-encoder/stsb-distilroberta-base" "cross-encoder/stsb-roberta-base" "cross-encoder/stsb-roberta-large" "cross-encoder/quora-distilroberta-base" "cross-encoder/quora-roberta-base" "cross-encoder/quora-roberta-large" "cross-encoder/nli-deberta-v3-base" "cross-encoder/nli-deberta-base" "cross-encoder/nli-deberta-v3-xsmall" "cross-encoder/nli-deberta-v3-small" "cross-encoder/nli-roberta-base" "cross-encoder/nli-MiniLM2-L6-H768" "cross-encoder/nli-distilroberta-base")
declare -a finetuned_models=("all-distilroberta-v1" "multi-qa-mpnet-base-cos-v1" "paraphrase-multilingual-mpnet-base-v2" "distiluse-base-multilingual-cased-v1" "paraphrase-multilingual-MiniLM-L12-v2")

for k_fold in "${k_folds[@]}"
do
    echo "FOLD_${k_fold}"
       
    # Bi-Encoder models evaluation
    for model in "${biencoder_models[@]}"
    do
	echo "- Bi-Encoder [ ${model} ]"
	python src/TRiC-sBERT-BiEncoder.py --model "${model}" --batch_size "${batch_size}" --device cuda --k_fold "${k_fold}"
    done

    # Crossencoder models evaluation
    for model in "${crossencoder_models[@]}"
    do
	echo "- CrossEncoder [ ${model} ]"
	python src/TRiC-sBERT-CrossEncoder.py --model "${model}" --batch_size "${batch_size}" --device cuda --k_fold "${k_fold}"
    done
    
    
    # Finetuned models evaluation
    for model in "${finetuned_models[@]}"
    do
	echo "- Finetuned [ ${model} ]"
	# Replace model_2e-06_0.0 with best parameters
	python src/TRiC-sBERT-BiEncoder.py --model "models/${model}_${k_fold}/model_2e-06_0.0" --batch_size "${batch_size}" --device cuda --k_fold "${k_fold}"
    done
    
done

python TRaC-SBert-BiEncoder.py --model all-distilroberta-v1 #adr
python TRaC-SBert-BiEncoder.py --model distiluse-base-multilingual-cased-v1 #dbm
python TRaC-SBert-BiEncoder.py --model paraphrase-multilingual-mpnet-base-v2 #par
python TRaC-SBert-BiEncoder.py --model paraphrase-multilingual-MiniLM-L12-v2 #pam
python TRaC-SBert-BiEncoder.py --model multi-qa-mpnet-base-cos-v1 #mqa
