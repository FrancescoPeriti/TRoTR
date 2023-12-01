import numpy as np

model2results = {}
metrics = []
i2k = {}

with open('/mimer/NOBACKUP/groups/cik_data/NAACL24/TRiC-stats.tsv') as f:
    for j,line in enumerate(f):
        line = line[:-1].split('\t')
        if not j == 0:
            D = {}
            for i,k in enumerate(line):
                D[i2k[i]] = line[i]

            model = D['model']
            fold = int(D['k_fold'])
            del D['model']
            del D['k_fold']

            if not model in model2results:
                model2results[model] = {}

            model2results[model][fold] = {}

            for k in D:
                if j == 1:
                    metrics.append(k)
                model2results[model][fold][k] = float(D[k])
        else:
            for i,k in enumerate(line):
                i2k[i] = k


modelstring2config = {}
maskmodels = {}

with open('TRiC-stats.tsv') as f:
    for j,line in enumerate(f):
        line = line[:-1].split('\t')
        if not j == 0:
            D = {}
            for i,k in enumerate(line):
                D[i2k[i]] = line[i]

            model = D['model']
            fold = int(D['k_fold'])
            del D['model']
            del D['k_fold']

            modelsplit = model.split('_')

            if len(modelsplit) > 2:
                lr = modelsplit[1]
                if modelsplit[-1] == 'mask':
                    maskmodels[model] = (modelsplit[0], lr)
                else:
                    modelstring2config[model] = (modelsplit[0], lr)

            if not model in model2results:
                model2results[model] = {}

            model2results[model][fold] = {}

            for k in D:
                model2results[model][fold][k] = float(D[k])
        else:
            for i,k in enumerate(line):
                i2k[i] = k


model2metrics = {}
best_config = {}
dev_idx = metrics.index('dev-spearman_corr')
model2best = {}
model2score = {}

for model in model2results:
    metrics_mean = [np.mean([model2results[model][k][metric] for k in range(1,11)]) for metric in metrics]
    metrics_std = [np.std([model2results[model][k][metric] for k in range(1,11)]) for metric in metrics]
    model2metrics[model] = {'mean':metrics_mean, 'std':metrics_std}
    if model in modelstring2config:
        dev_score = metrics_mean[dev_idx]
        if modelstring2config[model][0] in model2best:
            if dev_score > model2score[modelstring2config[model][0]]:
                model2best[modelstring2config[model][0]] = model
                model2score[modelstring2config[model][0]] = dev_score
        else:
            model2best[modelstring2config[model][0]] = model
            model2score[modelstring2config[model][0]] = dev_score

for model in list(model2metrics.keys()):
    if model in maskmodels:
        if not maskmodels[model][1] == modelstring2config[model2best[maskmodels[model][0]]][1]:
            del model2metrics[model]

with open('TRiC-stats_agg.tsv','w+') as f:
    header = []
    for i in range(len(i2k)):
        if i2k[i] == 'k_fold':
            continue
        elif i2k[i] == 'model':
            header.append('model')
        else:
            header.append(i2k[i])
            header.append(i2k[i]+'_std')

    f.write('\t'.join(header) + '\n')
    for model in model2metrics:
        if not model in modelstring2config or (model in modelstring2config and model2best[modelstring2config[model][0]] == model):
            strline = [model]
            for j in range(len(model2metrics[model]['mean'])):
                strline.append(str(model2metrics[model]['mean'][j]))
                strline.append(str(model2metrics[model]['std'][j]))
            line = '\t'.join(strline)
            f.write(line+'\n')