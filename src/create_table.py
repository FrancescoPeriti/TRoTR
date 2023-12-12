import numpy as np

model2results = {}
metrics = []
i2k = {}

with open('TRiC-stats_agg.tsv') as f:
    for j,line in enumerate(f):
        line = line[:-1].split('\t')
        if not j == 0:
            D = {}
            for i,k in enumerate(line):
                D[i2k[i]] = line[i]

            model = D['model']
            del D['model']

            if not model in model2results:
                model2results[model] = {}

            for k in D:
                if j == 1:
                    metrics.append(k)
                model2results[model][k] = float(D[k])
        else:
            for i,k in enumerate(line):
                i2k[i] = k

"""
basemodel2conf = {}
for model in model2results:
    modelsplit = model.split('_')
    if len(modelsplit) > 2:
        lr = modelsplit[1]
        if modelsplit[-1] == 'mask':
            maskmodels[model] = (modelsplit[0], lr)
        else:
            modelstring2config[model] = (modelsplit[0], lr)
"""

base_models = ['all-distilroberta-v1', 'all-MiniLM-L12-v2', 'multi-qa-mpnet-base-cos-v1', 'paraphrase-multilingual-mpnet-base-v2']

model2lines = {}
model2bestconfig = {}
        
for model in model2results:

    base_model = model.split('_')[0]
    if len(model.split('_')) > 2:
        model2bestconfig[base_model] = model.split('_')[1]+'_'+model.split('_')[2]

    if base_model in base_models:
        scores = []

        for split in ['test','test.oov']:
            for label in ['_label0','_label1','']:
                if not label == '':
                    precision = model2results[model][f'{split}-precision{label}']
                    precision_std = model2results[model][f'{split}-precision{label}_std']
                    recall = model2results[model][f'{split}-recall{label}']
                    recall_std = model2results[model][f'{split}-recall{label}_std']
                    f1 = model2results[model][f'{split}-f1_scores{label}']
                    f1_std = model2results[model][f'{split}-f1_scores{label}_std']
                    scores.append("{:.2f}".format(precision)[1:] + '±' + "{:.2f}".format(precision_std)[1:])
                    scores.append("{:.2f}".format(recall)[1:] + '±' + "{:.2f}".format(recall_std)[1:])
                    scores.append("{:.2f}".format(f1)[1:] + '±' + "{:.2f}".format(f1_std)[1:])
                else:
                    f1 = model2results[model][f'{split}-f1_score']
                    f1_std = model2results[model][f'{split}-f1_score_std']
                    scores.append("{:.2f}".format(f1)[1:] + '±' + "{:.2f}".format(f1_std)[1:])

            spearman = model2results[model][f'{split}-spearman_corr']
            spearman_std = model2results[model][f'{split}-spearman_corr_std']
            scores.append("{:.2f}".format(spearman)[1:] + '±' + "{:.2f}".format(spearman_std)[1:])

        line = '\t'.join(scores)
        model2lines[model] = line

with open('results.tsv','w+') as f:
    for model in base_models:
        f.write(model + '\t' + model2lines[model])
        f.write('\n')
        f.write('+FT' + '\t' +model2lines[model+'_'+model2bestconfig[model]])
        f.write('\n')
        f.write('+MASK' + '\t' + model2lines[model+'_mask'])
        f.write('\n')
        f.write('+FT+MASK' + '\t' +model2lines[model + '_' + model2bestconfig[model]+'_mask'])
        f.write('\n')
