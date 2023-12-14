declare models=("all-distilroberta-v1" "multi-qa-mpnet-base-cos-v1" "paraphrase-multilingual-mpnet-base-v2" "distiluse-base-multilingual-cased-v1" "paraphrase-multilingual-MiniLM-L12-v2")
declare folds=(1 2 3 4 5 6 7 8 9 10)
declare lrs=(1e-6 2e-6 5e-6 1e-5 2e-5)


for model in "${models[@]}";
do
	for fold in "${folds[@]}";
	do
		for lr in "${lrs[@]}";
		do
			python3 baseline_sbert.py --train_path='TRoTR/datasets/FOLD_'$fold'/pair-by-line/train.binary.jsonl' --dev_path='TRoTR/datasets/FOLD_'$fold'/pair-by-line/dev.oov.ranking.jsonl' --loss 'contrastive' --finetune_sbert --model_type "siamese" --sbert_pretrained_model $model --evaluation "correlation" --lr $lr --weight_decay 0 --output_path "models/"$model'_'$fold
		done
	done
done





