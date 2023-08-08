# B-CANF: Adaptive B-frame Coding with Conditional Augmented Normalizing Flows
Accpeted to TCSVT 2023

This repository contains the source code of our TCSVT 2023 paper **B-CANF** ([arXiv](https://arxiv.org/abs/2209.01769)).

## Abstract
>Over the past few years, learning-based video compression has become an active research area. However, most works focus on P-frame coding. 
Learned B-frame coding is under-explored and more challenging. This work introduces a novel B-frame coding framework, termed B-CANF, 
that exploits conditional augmented normalizing flows for B-frame coding. B-CANF additionally features two novel elements: frame-type adaptive coding and B*-frames. 
Our frame-type adaptive coding learns better bit allocation for hierarchical B-frame coding by dynamically adapting the feature distributions according to
the B-frame type. Our B*-frames allow greater flexibility in specifying the group-of-pictures (GOP) structure by reusing the B-frame codec to mimic P-frame coding, 
without the need for an additional, separate P-frame codec.

## Install

```bash
conda env create -f ./requirements/environment.yml
bash install.sh
```

## Pre-trained Weights
|     Metrics    |   Low Rate   $\to$  High Rate     |
|:--------------:|:---------------------------------:|
|      PSNR      |     [1](), [2](), [3](), [4]()    |
|     MS-SSIM    |     [1](), [2](), [3](), [4]()    |

## Example Usage
Specify run mode, checkpoint, source video* path, save directory, num_frames, coding intra-period and gop size accordingly.
* Source video: a folder contains frames named frame_1.png~frame_XXX.png


### Test Mode
* Usage: export coding R-D csv file.
* Command: `python test.py --mode test --ckpt ./ckpts/psnr_3.pth --src ./demo_seq --save_dir ./result --num_frames 97 --intra_period 32 --gop 16`

### Compress
* Usage: export coding R-D csv file and compressed bin file.
* Command: `python test.py --mode compress --ckpt ./ckpts/psnr_3.pth --src ./demo_seq --save_dir ./result --num_frames 97 --intra_period 32 --gop 16`

### Decompress
* Usage: export reconstructed video
* Command: `python test.py --mode decompress --ckpt ./ckpts/psnr_3.pth --src ./result/demo_seq.bin --save_dir ./result`


## Citation
If you find our project useful, please cite the following paper.
```
@ARTICLE{10201921,
  author={Chen, Mu-Jung and Chen, Yi-Hsin and Peng, Wen-Hsiao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={B-CANF: Adaptive B-frame Coding with Conditional Augmented Normalizing Flows}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3301016}}
```