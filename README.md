# Multimodal Pretraining of Medical Time Series and Notes

This is the official code for "Multimodal Pretraining of Medical Time Series and Notes" at Machine Learning for Health 2023. The paper can be found at the following link: [https://arxiv.org/abs/2312.06855](https://arxiv.org/abs/2312.06855)

## Data

### MIMIC-III

The dataset used for this paper is MIMIC-III. The data can be downloaded here [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/). **NOTE**: To gain access to this dataset, you will need to complete the required training. 

### MIMIC-III Benchmark

Once you've downloaded the MIMIC-III dataset, you will need to build the MIMIC-III Benchmark from [https://github.com/YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks). We used a modified version of this code so that we can index patient IDs and read in and out times without opening files. Replace:

```
mimic3-benchmarks/mimic3benchmark/scripts/extract_episodes_from_subjects.py
mimic3-benchmarks/mimic3benchmark/scripts/create_decompensation.py
mimic3-benchmarks/mimic3benchmark/scripts/create_in_hospital_mortality.py
mimic3-benchmarks/mimic3benchmark/scripts/create_length_of_stay.py
mimic3-benchmarks/mimic3benchmark/scripts/create_phenotyping.py
mimic3-benchmarks/mimic3benchmark/scripts/create_multitask.py
```

with:

```
multimodal-medical-pretraining/mimic3benchmark/extract_episodes_from_subjects.py
multimodal-medical-pretraining/mimic3benchmark/create_decompensation.py
multimodal-medical-pretraining/mimic3benchmark/create_in_hospital_mortality.py
multimodal-medical-pretraining/mimic3benchmark/create_length_of_stay.py
multimodal-medical-pretraining/mimic3benchmark/create_phenotyping.py
multimodal-medical-pretraining/mimic3benchmark/create_multitask.py
```

Once you've replaced that file, build the benchmarks as described here: [https://github.com/YerevaNN/mimic3-benchmarks/tree/master#building-the-benchmark](https://github.com/YerevaNN/mimic3-benchmarks/tree/master#building-the-benchmark).

For our semi-supervised experiments, we created new listfiles which can be downloaded [here](https://drive.google.com/drive/folders/1wB-4kUrNB9cHqD1qvR5fFEOaIUXmXTxI?usp=sharing). These listfiles need to be placed in the root directory of your MIMIC-III Benchmark data.

After adding the files, the structure of your MIMIC-III Benchmark folder should look like this:

Structure for ImageNet Data
```
mimic3-benchmarks
├── phenotyping
│   ├── 1percent_train_listfile.csv
│   ├── 10percent_train_listfile.csv
│   ├── 50percent_train_listfile.csv
│   ├── 1percent_val_listfile.csv
│   ├── 10percent_val_listfile.csv
│   ├── 50percent_val_listfile.csv
│   ├── train_listfile.csv
│   ├── val_listfile.csv
│   ├── test_listfile.csv
│   └── 
├── in-hospital-mortality
│   ├── 1percent_train_listfile.csv
│   ├── 10percent_train_listfile.csv
│   ├── 50percent_train_listfile.csv
│   ├── 1percent_val_listfile.csv
│   ├── 10percent_val_listfile.csv
│   ├── 50percent_val_listfile.csv
│   ├── train_listfile.csv
│   ├── val_listfile.csv
│   ├── test_listfile.csv
│   └── 
├── root
│   ├── 
│   └── 
```

### Requirements

The python version used to run our experiments is 3.9.16. Requirements can be found in the requirements.txt file. Install them by running:

`pip install -r requirements.txt`

## Pretraining

Pretraining can be run using the script located at `experiments/measurement_notes/measurement_notes_pretraining.py`. We've included an example command in `pretrain`

## Finetune

Pretraining can be run using the script located at `experiments/measurement_notes/measurement_notes_downstream.py`. We've included an example command in `finetuning`. This command requires an experiment and task be provided. Possible experiments include `semi_0_5_eval`, `semi_0_1_eval`, `semi_0_01_eval`, `full_eval`, and `linear_eval`. Possible tasks include: `IHM` and `Phenotyping`. A pretrained model can be provided using `pretrained_path`. An example of our finetuning experiment can be found at `linear_eval`. Download the model used for this evaluation [here](https://drive.google.com/drive/folders/1wB-4kUrNB9cHqD1qvR5fFEOaIUXmXTxI?usp=sharing)
