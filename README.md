# Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-XXXX-B21A1B)](https://github.com/shikiw/Modality-Integration-Rate)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-blue)](https://github.com/shikiw/Modality-Integration-Rate)
[![GitHub Stars](https://img.shields.io/github/stars/shikiw/Modality-Integration-Rate?style=social)](https://github.com/shikiw/Modality-Integration-Rate/stargazers)


This repository provides the official PyTorch implementation of the following paper: 
> [**Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate**](https://github.com/shikiw/Modality-Integration-Rate) <br>
> [Qidong Huang](https://shikiw.github.io/)<sup>1,2</sup>, 
> [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en)<sup>2,3</sup>, 
> [Pan Zhang](https://panzhang0212.github.io/)<sup>2</sup>,
> [Yuhang Zang](https://yuhangzang.github.io/) <sup>2</sup>,
> [Yuhang Cao](https://scholar.google.com/citations?user=sJkqsqkAAAAJ&hl=zh-CN) <sup>2</sup>, 
> [Jiaqi Wang](https://myownskyw7.github.io/)<sup>2</sup>,
> [Dahua Lin](http://dahua.site/)<sup>2</sup>, 
> [Weiming Zhang](http://staff.ustc.edu.cn/~zhangwm/index.html)<sup>1</sup>, 
> [Nenghai Yu](https://scholar.google.com/citations?user=7620QAMAAAAJ&hl=en)<sup>1</sup> <br>
> <sup>1</sup>University of Science and Technology of China, <sup>2</sup>Shanghai AI Laboratory, <sup>3</sup>The Chinese University of Hong Kong <br>

## 🎯 News

**[2024.10.10]** 🚀 We release the paper at [ArXiv]() and [HuggingFace]()!



## ⭐️ TL;DR
### 1. For MIR
If you just want to use MIR as the pre-training indicator of your own model, no additional environment is required.

1. Ensure the packages such as ```torch```, ```numpy```, and ```scipy``` are installed.
2. Replace the model preprocessing and generation in ```mir.py``` with your own model's code, we display LLaVA's code as the reference.
3. Specify the input args and run the command:
```
python mir.py --model_path PATH/TO/MODEL --base_llm PATH/TO/LLM --text_data_path PATH/TO/TEXT/DATA --image_data_path PATH/TO/VISION/DATA --eval_num 100 --mode fast
```
Note that ```base_llm``` is not required if you haven't train the base LLM during pre-training. 

You can also adjust the args to the intialization style of your model.

### 2. For MoCa
If you just want to use MoCa on your own model, we recommand you to following the steps below:

1. Adding (Here)



## Setup
The following steps are just listed for Linux. If you are using macOS or Windows, please refer to [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file).
1. Clone this repository and navigate to Modality-Integration-Rate folder
```
git clone https://github.com/shikiw/Modality-Integration-Rate.git
cd Modality-Integration-Rate
```
2. Install Package
```
conda create -n llava python=3.10 -y
conda activate llava
python -m pip install --upgrade pip  # enable PEP 660 support
python -m pip install -e .
```
3. Install additional packages for training cases
```
pythom -m pip install -e ".[train]"
pythom -m pip install flash-attn --no-build-isolation
```
