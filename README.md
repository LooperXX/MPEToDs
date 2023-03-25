# MPEToDs

<img src="imgs/SCIR_logo.png" style="width:40%; margin-right:20px;" /><img src="imgs/Westlake_logo.png" style="width:40%;" />

This repository contains the official `PyTorch` implementation of the paper:

[**Modularized Pre-training for End-to-end Task-oriented Dialogue.**](https://ieeexplore.ieee.org/abstract/document/10043710)

Libo Qin, Xiao Xu, Lehan Wang, Yue Zhang, Wanxiang Che.

IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)

If you use any source codes or the datasets included in this toolkit in your work, please cite the following paper. The bibtex are listed below:

```
@ARTICLE{qin-etal-2023-modularized,
  author={Qin, Libo and Xu, Xiao and Wang, Lehan and Zhang, Yue and Che, Wanxiang},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Modularized Pre-training for End-to-end Task-oriented Dialogue}, 
  year={2023},
  volume={},
  number={},
  pages={1-10},
  doi={10.1109/TASLP.2023.3244503}
}
```

## Architecture

![image](imgs/Architecture.png)

## Preparation

### Environment

Install conda and then create a new environment with our configuration file:

``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
bash Miniconda3-py37_4.12.0-Linux-x86_64.sh
conda env create -f environment.yml
```

We use [fitlog](https://github.com/fastnlp/fitlog) to track our experiments. [Here](https://fitlog.readthedocs.io/) is the documentation of fitlog (but only Chinese).
```bash
fitlog init
# you can use the below commands to view your experiment results. See https://fitlog.readthedocs.io/zh/latest/user/command_line.html for command line instructions.
fitlog log <log-dir>
```

### Data

All the data used in pre-training and fine-tuning are publicly available. Below are the links to the original data sources:
- [Schema](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
- [Taskmaster](https://github.com/google-research-datasets/Taskmaster)
- [MSR-E2E](https://github.com/xiul-msr/e2e_dialog_challenge/tree/master/data)
- [WOZ](https://github.com/nmrksic/neural-belief-tracker)
- [SMD](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)
- [CameRest676](https://github.com/yizhen20133868/Retriever-Dialogue)
- [MWOZ](https://github.com/budzianowski/multiwoz/)

You can download all data from [here](https://pan.baidu.com/s/1iimQ5GFKn6Wil5hogPQ5Wg?pwd=gi4k) and unzip them into the `data/` folder, which should have the following structure:

```
data
 ├── pre-train
 ├── fine-tune
 └── original
```

We also provide the pre-process scripts in `prepare_data/`:
- `preprocess_gp.py`: convert the data in `data/original/` folder into the format of Generation Module Pre-training (`data/pre-train/` provides the processed data)
- `augmentation_xxx.py`: convert the data in `data/fine-tune/` folder into the format of Knowledge-retriever Module Pre-training (`data/pre-train/` provides the processed data)

### Checkpoints
You can download all saved checkpoints from [pre-train](https://pan.baidu.com/s/1edEzo3mWCoriWzWl_rOwuQ?pwd=7mnh) and [fine-tune](https://pan.baidu.com/s/1mVYk9EuHUZS6VErYX8u6_A?pwd=km2t) . Then, please unzip them into the `save/` folder, which should have the following structure:

```
save
 ├── fine-tune
 │   ├── SMD_Best
 │   ├── WOZ_Best
 │   └── CAM_Best
 └── pre-train
     ├── GP
     └── KRP
```


### Pre-trained Models

In this paper, we use [GPT-2](https://huggingface.co/gpt2) or [DialoGPT-Medium](https://huggingface.co/microsoft/DialoGPT-medium) to initialize our generation module. 
If you want to re-pre-train generation module, please download them from [here](https://pan.baidu.com/s/11868edKhIM1l_AMpMZzTCA?pwd=97jh) and unzip them into the `pre-train/` folder, which should have the following structure:

```
pre-train
 ├── medium_ft.pkl
 ├── pytorch_model.bin
 ├── merges.txt
 ├── vocab.json
 └── config.json
```

## How to Run it

```bash
conda activate MPEToDs
# 1. Pre-train Generation Module
python pretrain_GPT.py -g -uf -fld=logs/GP -bsz=32 -accs=2 -gl=5e-05 --warmup_steps=2500 --total_steps=110000 --valid_steps=2000 --logging_steps=200 
# 2. Pre-train Knowledge-retriever Module
python pretrain_KB.py -g -uf -fg -ds=smd -fld=logs/KRP_SMD -pgpt=save/pre-train/GP -bsz=32 -accs=2 -dr=0.1 -hdd=256 -lr=0.001 --warmup_steps=250 --total_steps=15000 --valid_steps=100 --logging_steps=10
python pretrain_KB.py -g -uf -fg -ds=cam -fld=logs/KRP_CAM -pgpt=save/pre-train/GP -bsz=32 -accs=1 -dr=0.2 -hdd=128 -lr=0.001 --warmup_steps=400 --total_steps=10000 --valid_steps=100 --logging_steps=8
python pretrain_KB.py -g -uf -fg -ds=woz -fld=logs/KRP_WOZ -pgpt=save/pre-train/GP -bsz=32 -accs=1 -dr=0.1 -hdd=128 -lr=0.001 --warmup_steps=1600 --total_steps=40000 --valid_steps=200 --logging_steps=20
# 3. Fine-tune
python fine_tune.py -g -uf -ft -ds=smd -fld=logs/fine_tune_SMD -pa=3 -pgpt=save/pre-train/GP -pkb=save/pre-train/KRP/SMD -bsz=16 -accs=1 -dr=0.1 -hdd=256 -lr=0.0007 -gl=4e-05 --warmup_steps=500 --total_steps=4000 --logging_steps=4
python fine_tune.py -g -uf -ft -ds=cam -fld=logs/fine_tune_CAM -pa=3 -pgpt=save/pre-train/GP -pkb=save/pre-train/KRP/CAM -bsz=16 -accs=1 -dr=0.1 -hdd=128 -lr=0.001 -gl=4e-05 --warmup_steps=60 --total_steps=1050 --logging_steps=4
python fine_tune.py -g -uf -ft -ds=woz -fld=logs/fine_tune_CAM -pa=3 -pgpt=save/pre-train/GP -pkb=save/pre-train/KRP/WOZ -bsz=4 -accs=4 -dr=0.1 -hdd=128 -lr=0.001 -gl=7e-05 --warmup_steps=500 --total_steps=5332 --logging_steps=5
```

**Config Notes:**

1. The actual batch size = `bsz` * `accs`, *i.e.*, batch_size and accumulation_steps.
2. We valid the model every `valid_steps` in pre-training and every epoch in fine-tuning.
3. We fine-tune the model for 10 epochs in our experiments. You can change the `total_steps` to control the number of epochs. 
4. `total_steps`, `warmup_steps` should change with `bsz` and `accs`. Take `Fine-tune of SMD` as an example, if you change `bsz` from `16` to `32` and `accs` from `1` to `3`, you should change `total_steps` from `4000` to `4000/2/3`, and change `warmup_steps` from `500` to `500/2/3`.

## Acknowledgement

We are highly grateful for the public code of the following papers, our code is partly based on them:

- **DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation.**

    Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.

    ACL 2020 Demo. [[Paper]](https://arxiv.org/abs/1911.00536) [[Code]](https://github.com/microsoft/DialoGPT)


- **Global-to-local Memory Pointer Networks for Task-Oriented Dialogue.**

    Chien-Sheng Wu, Richard Socher, Caiming Xiong.

    ICLR 2019. [[Paper]](https://arxiv.org/abs/1901.04713) [[Open Reivew]](https://openreview.net/forum?id=ryxnHhRqFm) [[Code]](https://github.com/jasonwu0731/GLMP)


- **Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog.**

    Libo Qin, Xiao Xu, Wanxiang Che, Yue Zhang, Ting Liu.

    ACL 2020. [[Paper]](https://arxiv.org/abs/2004.11019) [[Code]](https://github.com/LooperXX/DF-Net)
