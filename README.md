> **Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection**  
> Zhijing Wan, Zhixiang Wang, Zheng Wang, Xin Xu, Shin'ichi Satoh.
> 
> ICML 2025
>
> [arxiv](https://arxiv.org/pdf/2506.14473)

## 📌 Overview
This repository contains the official PyTorch implementation of **RAM-APL**, along with several competitive baselines such as MIN, kCenterGreedy, and Moderate-DS. It supports both traditional pre-trained models (e.g., the target model trained on the target dataset) and foundation models as feature extractors (e.g., CLIP, DINOv2, SigLIP, EVA-CLIP).

## 🔧 Installation

Make sure to install dependencies:

```bash
pip install -r requirements.txt
```

## 🛆 Foundation Model Weights (Important)

By default, foundation model weights (e.g., DINOv2, CLIP, SIGLIP, EVA-CLIP) are **loaded offline** from local directories.

It is recommended that you manually store the pre-trained weights under:

```bash
./deepcore/methods/pretrain/
```

To enable **online loading** (via HuggingFace or TorchHub), please refer to the file:

```bash
./deepcore/methods/earlytrain.py
```

Set your `AutoModel` / `torch.hub.load` methods accordingly, or configure your internet-enabled environment to fetch weights dynamically.

---

## 🧪 Example Usage

We provide sample commands for evaluating different subset selection methods under varying settings.

### ➔ 1. Single-model Study: MIN method on Oxford-IIITPet with 20% symmetric label noise

#### a) With pretraining (Traditional IE):

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py \
  --fraction 0.5 \
  --dataset Pet_NOISY --noise_type symmetric --noise_rate 0.2 \
  --data_path /path/to/data \
  --num_exp 3 --workers 4 --optimizer SGD -se 10 \
  --selection MIN --model ResNet18 --lr 0.1 \
  -sp ./results/MIN_pet_sym0.2_10epoch_0.5 \
  --batch 128 >> ./results/MIN_pet_sym0.2_10epoch_0.5.txt 2>&1
```

#### b) With TinyImageNet pretraining:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py \
  --fraction 0.5 \
  --dataset_pretrain TinyImageNet --dataset Pet_NOISY \
  --noise_type symmetric --noise_rate 0.2 \
  --data_path /path/to/data \
  --num_exp 3 --workers 4 --optimizer SGD -se 10 \
  --selection MIN --model ResNet18 --lr 0.1 \
  -sp ./results/MIN_pet_sym0.2_TIN_10epoch_0.5 \
  --batch 128 >> ./results/MIN_pet_sym0.2_TIN_10epoch_0.5.txt 2>&1
```

#### c) With DINOv2 feature extraction (zero pretrain epochs):

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py \
  --specific_model DINOV2 \
  --fraction 0.5 \
  --dataset Pet_NOISY --noise_type symmetric --noise_rate 0.2 \
  --data_path /path/to/data \
  --num_exp 3 --workers 4 --optimizer SGD -se 0 \
  --selection MIN --model ResNet18 --lr 0.1 \
  -sp ./results/MIN_pet_sym0.2_DINOv2_0.5 \
  --batch 128 >> ./results/MIN_pet_sym0.2_DINOv2_0.5.txt 2>&1
```

---

### ➔ 2. RAM-APL on Pet dataset

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py \
  --fraction 0.5 \
  --dataset Pet \
  --data_path /path/to/data \
  --num_exp 5 --workers 4 --optimizer SGD -se 0 \
  --selection RAM_APL --model ResNet18 --lr 0.1 \
  -sp ./results/DINOv2Clip_mcr_a0.2k1_0.5 \
  --batch 128 >> ./results/DINOv2Clip_mcr_a0.2k1_0.5.txt 2>&1
```

---

## 🗂 Project Structure (partial)

```bash
.
├── main.py                # Main entry point
├── utils.py/              # Training and testing functions
├── deepcore/
│   ├── methods/           # All subset selection methods
│   │   ├── min.py
│   │   ├── ram_apl.py
│   │   ├── kcentergreedy.py
│   │   └── moderate_ds.py
│   │   ├── pretrain/      # Downloaded foundation model weights (offline mode)
│   ├── nets/              # Model definitions and wrappers
│   ├── datasets/          # Dataset loaders
├── results/               # Saved logs and outputs
└── requirements.txt       # Dependencies
```

---

## 📌 Notes

* All dataset paths are expected to be provided via `--data_path`.
* We support training with or without pretraining, as well as using foundation models as frozen feature extractors.
* To enable offline loading of foundation model weights (default), please download and store them under `deepcore/methods/pretrain/`.
* To enable online loading, refer to the logic in `deepcore/methods/earlytrain.py`.
* If you need help preparing your datasets or pretrained weights, feel free to open an issue (if public).

---

## 📜 License

This codebase is for academic research purposes only.

## 📬 Bibtex
If you make use of our work, please consider to cite:

```bibtex
@article{wan2025foundation,
  title={Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection},
  author={Wan, Zhijing and Wang, Zhixiang and Wang, Zheng and Xu, Xin and Satoh, Shin'ichi},
  journal={arXiv preprint arXiv:2506.14473},
  year={2025}
}
```

## Credits
The implementation is based on [DeepCore](https://github.com/PatrickZH/DeepCore) code. Thanks for their brilliant work!
