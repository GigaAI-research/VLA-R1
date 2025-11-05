# <img src="https://github.com/GigaAI-research/VLA-R1/blob/website/assets/vlar1_logo.png" alt="logo" width="50"/> VLA-R1: Enhancing Reasoning in Vision-Language-Action Models

This is the official repository for the paper:
> **VLA-R1: Enhancing Reasoning in Vision-Language-Action Models**
>
> Angen Ye\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*, Boyuan Wang, Xiaofeng Wang, Dapeng Zhang, and [Zheng Zhu](http://www.zhengzhu.net/)<sup>†</sup>
>
> \*Equal contribution. <sup>†</sup>Corresponding author.
>
> ### [Paper](https://arxiv.org/abs/2510.01623) | [Website](https://gigaai-research.github.io/VLA-R1) | [Data]() | [Models]() | [HF Paper](https://huggingface.co/papers/2510.01623)

https://github.com/user-attachments/assets/df8c2ae5-59e7-4119-8bb7-6712fcd93246

## Create environment using Conda
```
conda env create -f environment.yml
conda activate vla_r1
```
## Model

huggingface-cli download --repo-type model --resume-download GigaAI-Research/vla-r1

## Dataset

huggingface-cli download --repo-type dataset --resume-download GigaAI-Research/vla_r1

## Training
```
bash RFT_training/train_utils/run_vla_r1_3b.sh
```

## Inference
```
python scripts/server.py
python scripts/inference.py
```

## ✏️ Citation
If you find our code or paper helpful, please consider starring ⭐ us and citing:
```bibtex
@article{ye2025vlar1,
  title={VLA-R1: Enhancing Reasoning in Vision-Language-Action Models},
  author={Ye, Angen and Zhang, Zeyu and Wang, Boyuan and Wang, Xiaofeng and Zhang, Dapeng and Zhu, Zheng},
  journal={arXiv preprint arXiv:2510.01623},
  year={2025}
}
```







