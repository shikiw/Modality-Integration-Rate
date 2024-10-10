# Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2410.07167-B21A1B)](https://arxiv.org/abs/2410.07167)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-blue)](https://huggingface.co/papers/2410.07167)
[![GitHub Stars](https://img.shields.io/github/stars/shikiw/Modality-Integration-Rate?style=social)](https://github.com/shikiw/Modality-Integration-Rate/stargazers)


This repository provides the official PyTorch implementation of the following paper: 
> [**Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate**](https://arxiv.org/abs/2410.07167) <br>
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

**[2024.10.10]** 🚀 We release the paper at [ArXiv](https://arxiv.org/abs/2410.07167) and [HuggingFace](https://huggingface.co/papers/2410.07167)!

**[2024.10.10]** 🚀 This project page has been built!

## 👨‍💻 Todo

- [x] Release the code of MIR
- [x] Release the training code and evaluation code of MoCa
- [ ] Release the data and checkpoints 



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

1. Copy the code of [MoCa module](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/transformers-4.37.2/src/transformers/models/llama/modeling_llama.py#L122-L139) into the modeling code of your own model and ensure MoCa is equipped by the base LLM layer in both [initialization](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/transformers-4.37.2/src/transformers/models/llama/modeling_llama.py#L809-L814) and [forward](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/transformers-4.37.2/src/transformers/models/llama/modeling_llama.py#L868-L870) functions.
2. Make sure that the input preprocessing can compute the ```modality_mask```, please refer to [Line183-184](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/llava_arch.py#L183-L184), [Line269-276](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/llava_arch.py#L269-L276) and [Line373-382](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/llava_arch.py#L373-L382) in ```llava/model/llava_arch.py```. Also, make sure that the ```modality_mask``` can be successsfully delivered into the model forward pass, e.g., adding it as the formal parameter of each forward function, like [Line70](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L70), [Line88](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L88), [Line96](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L96), [Line106](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L106), [Line127](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L127), [Line137](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L137), [Line145](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L145), [Line157](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L157), [Line166](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L166), [Line174-175](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L174-L175) in ```llava/model/language_model/llava_llama.py```. 
3. Check some details to support the usage of ```use_moca=True```, such as (it is recommanded to search ```use_moca``` in this repo to find which places should be revised):
   1）Add it into the model config ([here](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/model/language_model/llava_llama.py#L35)).
   2) Add it into training arguments ([here](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/train/train.py#L72)).
   3) Unlock it during training ([here](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/train/train.py#L1056-L1060)).
   4) Ensure the correct checkpoint saving ([here1](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/train/train.py#L199), [here2](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/train/llava_trainer.py#L278), [here3](https://github.com/shikiw/Modality-Integration-Rate/blob/501d64dd37aa5382caf97d14c1da9b088bb8b4c7/llava/train/llava_trainer.py#L299)).
4. Add ```--use_moca``` when running the training command to enable the usage of MoCa.



## 📜 Setup
If you want to use our codebase (modified on LLaVA) for reproduction, you are recommanded to build a new environment though the steps below. 
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
python -m pip install -e transformers-4.37.2
```
3. Install additional packages for training cases
```
pythom -m pip install -e ".[train]"
pythom -m pip install flash-attn --no-build-isolation
```


## MIR

To reproduce the MIR implementation on this codebase, you can follow these steps:
1. Specify the ```text_data_path``` and ```image_data_path``` for MIR calculation. You can also specify them like [Line55-64](https://github.com/shikiw/Modality-Integration-Rate/blob/b9ec4d3b080444dcf2b2b7cc3d21a3fdb9dcb42b/mir.py#L55-L64) in ```mir.py```, using TextVQA val images and CNN/DM text by default, i.e., 
   1) Download [TextVQA_0.5.1_val.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to ```PATH/TO/VISION/DATA```.
   2) Download [CNN stories](https://cs.nyu.edu/~kcho/DMQA/) and extract to ```PATH/TO/TEXT/DATA```.
   3) Modify [Line55-64](https://github.com/shikiw/Modality-Integration-Rate/blob/b9ec4d3b080444dcf2b2b7cc3d21a3fdb9dcb42b/mir.py#L55-L64) with the text data path and image data path.
2. If you pre-train only MLP, run this command:
```
python mir.py --model_path PATH/TO/MODEL --base_llm PATH/TO/LLM --eval_num 100 --mode fast
```
3. If your pre-train any part of ViT or base LLM, run this command:
```
python mir.py --model_path PATH/TO/MODEL --eval_num 100 --mode fast
```

## Train
This codebase is based on [LLaVA](https://github.com/haotian-liu/LLaVA) and [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V), where we introduce some new features and now it supports the following inputs in the launch script:
   1) ```--tune_vision_tower``` and ```--tune_vit_from_layer```
   2) ```--tune_language_model``` and ```--tune_llm_utill_layer```
   3) ```--tune_entire_model```
   4) ```--data_scale```
   5) ```--use_moca``` and ```--moca_std```

Some cases for reference: 

1. To pre-train the model with the customized data scale (e.g., 200K):
```
sh scripts/v1_5/pre_data_scale.sh
```

2. To pre-train the model (unlock the 13-24 layer of ViT and the 1-16 layer of base LLM), and SFT (unlock entire LLM by default):
```
sh scripts/v1_5/pre_unlock_vit-12_llm-16_sft.sh
```

3. To pre-train the model (unlock the 13-24 layer of ViT and the entire base LLM), and SFT (unlock entire LLM by default):
```
sh scripts/v1_5/pre_unlock_vit-12_llm-all_sft.sh
```

4. To apply MoCa in training:
```
sh scripts/v1_5/pre_sft_moca.sh
```


## Evaluation
We follow the original evaluation in [LLaVA](https://github.com/haotian-liu/LLaVA) for most of benchmarks. For [MMStar](https://github.com/MMStar-Benchmark/MMStar), we use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). 

See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). 


## Acknowledgement
This repo is based on the codebase of [LLaVA](https://github.com/haotian-liu/LLaVA) and [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V). Thanks for their impressive works!




