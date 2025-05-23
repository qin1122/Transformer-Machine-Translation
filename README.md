# Transformer-Machine-Translation

## 👀 Demos Video

### TODO

## 🔧 Dependencies and Installation
- Python >= 3.11
- [PyTorch >= 2.4.0](https://pytorch.org/)

### Installation
1. Clone repo

```bash
git clone https://github.com/qin1122/Transformer-Machine-Translation.git
cd Transformer-Machine-Translation
```

2. Install dependent packages (use conda)

```bash
conda create --name transformer
conda activate transformer
conda install python
pip install -r requirements.txt
```

## 🗂️ Datasets

I use a dataset consisting of 21,621 pairs of English and Chinese short sentences. Original dataset can be downloaded [here](https://pan.baidu.com/s/1Vb3PvFkfCvJ_JdapgEU_Hg?pwd=h43i).

I applied a better tokenizing method for Chinese sentences, which led to a higher BLEU score. The new dataset and the preprocessed original dataset can be downloaded [here](https://box.nju.edu.cn/d/6749c927a18847e584b7/)

## ⚙️ Train

Create a new yaml config file or use my config files (e.g. './configs/train_1.yaml'), put the yaml file in './configs'.

Then run:

```bash
python main.py --config 'configs/config.yaml'
```

You can monitor the training process in real time on [Weights & Biases (wandb)](https://wandb.ai).

> To use wandb, you have to login in terminal first, use the command below to login:

```bash
wandb login
```

## ⚡️ Quick Test

Create a new yaml config file or use my config files (e.g. './configs/test_origin.yaml'), put the yaml file in './configs'.

Then run:

```bash
python test.py --config 'configs/config.yaml'
```

## 🏰 Model Zoo

We conducted a total of 19 experiments, and the best model parameters and results can be downloaded [here](https://box.nju.edu.cn/d/f0333f5bddc74e17857a/).