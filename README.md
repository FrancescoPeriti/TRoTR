# TRoTR: Topic Relatedness of Text Reuse
This is the official repository for our paper _TRoTR: A Framework for Evaluating the Re-contextualization of Text Reuse_ .

Below, you will find instructions to reproduce our study. Feel free to contact us!

## Abstract
Computational approaches for _detecting_ text reuse do not focus on capturing the change between the original context of the reused text and their _re-contextualization_. In this paper, we rely on the notion of topic relatedness and propose a framework called __Topic Relatedness of Text Reuse (TRoTR)__ for evaluating the diachronic change of context in which text is reused. __TRoTR__ includes two NLP tasks: Text Reuse in-Context (TRiC) and Topic variation Ranking across Corpus (TRaC). TRiC is designed to evaluate the _topic relatedness_ between a pair of re-contextualizations. TRaC is designed to evaluate the overall topic variation within a set of re-contextualizations. We also provide a curated __TRoTR__ benchmark of biblical text reuse, human-annotated with topic relatedness. The benchmark exhibits a inter-annotator agreement of .811, calculated by average pair-wise correlation on assigned judgments. Finally, we evaluate multiple, established Sentence-BERT models on the __TRoTR__ tasks and find that they exhibit greater sensitivity to semantic similarity than topic relatedness. Our experiments show that fine-tuning these models can mitigate such a kind of sensitivity.

## Table of Contents

- [Getting Started](#getting-started)
- [Benchmark](#benchmark)
- [Evaluation setting](#evaluation-setting)
- [Evaluation results](#evaluation-results)
- [References](#references)

## Getting Started
Ensure you have met the following requirements:

- Python 3.10.4
- Required Python packages (listed in `requirements.txt`)

To install the required packages, you can use pip:

```bash
pip install -r requirements.txt
```

## Benchmark 
The ```TRoTR``` folder contains the data and labels of our benchmark.

- ```TRoTR/raw_data.jsonl``` contains the tweets manually collected by Twitter (now X).

#### Tutorial
We used the running version of the <a href='https://phitag.ims.uni-stuttgart.de/'>PhiTag annotation platform</a> to display the guidelines, tutorial and data to annotators.
Thus, our data adheres to the current format supported by PhiTag.

- ```TRoTR/tutorial``` contains the data used for training annotators in a 30-minute online session and their resulting judgments.

Notably, the tutorial data were excluded in our work.

#### Annotation guidelines
- ```TRoTR/guidelines.md``` contains the instructions followed by human annotators.

#### From raw_data.jsnol to PhiTag data
For each target quotation $t$, we randomly sampled of 150 unique context pairs $\langle t, c_1, c_2 \rangle$ without replacement from the full set of possible combinations. These were presented to annotators in randomized order to be judged for topic relatedness. 

To convert the ```TRoTR/raw_data.jsnol``` dataset into PhiTag format and implement random sampling of the context pairs $\langle t, c_1, c_2 \rangle$, we used the following command:

```
python src/random-sample.py
```

This creates the ```TRoTR/data``` folder. The folder contains an additional sub-folder for each quotation (e.g. ```TRoTR/data/(1 Corinthians 13 4)```). These subfolders contain the data in PhiTag format. For the sake of simplicity, we also created two file that contains all context usages (i.e., PhiTag uses) and context pairs (i.e., PhiTag instantes) used in our work:
- ```TRoTR/data/uses.csv```
- ```TRoTR/data/instances.csv```

We divided the annotation process into four distinct rounds, each covering a different set of targets. This division was implemented manually and only for the purpose of conducting a quality check between consecutive rounds during the annotation process. PhiTag uses and instances for each round can be found in folders ```TRoTR/rounds``` and ```TRoTR/judgments```, respectively.

After round 1, annotators were evaluated with a 30-minute online session on a subset of instances. We didn't consider these data in our benchmark. Anyway, you find the data in ```TRoTR/round/quality-check-1st-round.tsv``` and ```TRoTR/judgments/quality-check-1st-round.tsv```.

To join the uses and judgments files of different rounds into two comphrensive files, we used the following command:

```
python src/merge-round.py
```

These produces two comphrensive files:
- ```TRoTR/round/TRoTR.tsv``
- ```TRoTR/judgments/TRoTR.tsv``

#### Statistics
We computed inter-annotator agreements by using the ```stats+DURel.ipynb notebook```.

## Evaluation setting
##### TRiC
Our TRiC evaluation was conducted on 10 different Train-Test-Dev partitions. We can obtain the same 10 partitions by using the following command:

```
python src/cross_validation.py -s binary --n_folds 10
python src/cross_validation.py -s ranking --n_folds 10
```
This command results in 10 different sub-folders under the folder ```TRoTR/datasets/```. In particular, each folder contains the data in two formats: 
1. line-by-line: the contexts of a pair are represented one below the other on separate lines.
2. pair-by-line: the contexts of a pair are represented in the same line.

We used format 2, but we make available the same data also in the format 1.

Both format sub-folders contain the data for the Train-Test-Dev splits.

##### TRaC
Our TRaC evaluation was conducted on the full set of data. To generate ground truth data, we used the following command:

```
python src/topic_variation_scores.py
```

To generate benchmark data for TRaC, we used the following command:

```
python src/TRaC_dataset_generation.py
```

## Fine-tuning

To fine-tune the models used in our study, you can use the specific bash script:

```
sh finetuning.sh
```


## Evaluation result
For evaluation results, we used three scripts:
- ```src/TRiC-sBERT-BiEncoder.py```: for testing Bi-Encoder models on TRiC
- ```src/TRiC-sBERT-CrossEncoder.py```: for testing Cross-Encoder models on TRiC
- ```src/TRiC-sBERT-BiEncoder.py```: for testing Bi-Encoder models on TRaC

In particular, you can easily use the following bash command to call the three previous scripts and evaluate different models on both TRiC and TRaC.

```bash sequence-model-evaluation.sh```

This will create ```TRiC-stats.tsv``` and ```TRaC-stats.tsv``` files containing performance for different metrics and partitions.

## References
...
