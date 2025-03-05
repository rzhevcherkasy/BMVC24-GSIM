# GSIM 

This repository contains the official implementation and results for the BMVC 2024 paper:  
[Gaussian Splatting in Mirrors: Reflection-Aware Rendering via Virtual Camera Optimization](https://arxiv.org/pdf/2410.01614).  

## Data and Checkpoints  

Preprocessed datasets, checkpoints, and rendered images can be downloaded from  
[Google Drive](https://drive.google.com/drive/folders/1pLedNRYrhDk8LIvbX3-qfMd27wUplYXJ?usp=sharing)  

Compared to the paper version, we have fixed some code issues, leading to improvements in real-world scenarios. The released checkpoint results are shown below:  

| Scene  | PSNR  | SSIM  | LPIPS  |  
|--------|-------|-------|--------|  
| Living_room | 41.70 | 0.990 | 0.008  |  
| Office | 38.89 | 0.980 | 0.044  |  
| Washroom | 37.66 | 0.976 | 0.029  |  
| Market | 29.97 | 0.907 | 0.067  |  
| Lounge | 28.69 | 0.914 | 0.153  |  
| Discussion_room | 21.72 | 0.778 | 0.187  |  

## Environment Setup  

```sh
# Create and activate conda environment  
conda create -n GSIM python=3.8  
conda activate GSIM  

# Install PyTorch (CUDA 11.8)  
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia  

# Install dependencies (including rasterizer modified from [iComMa](https://github.com/YuanSun-XJTU/iComMa))  
pip install -r requirements.txt  
```

## Training  

Rename the corresponding scene configuration file:  
```sh
cp splatfacto_config_{scene}.py splatfacto_config.py
```
Then run training:  
```sh
python train.py -s <path to dataset> --eval  
```

## Rendering  

```sh
python render.py -m <path to checkpoint> --width <render width> --height <render height>  
```

## Metrics  

```sh
python metrics.py -m <path to checkpoint>  
```

## Limitations  

This work is an early 2024 exploration of 3D-GS in mirror reflections, with two main limitations:  

1. The most significant limitation, as shown in our paper, is that our model **only learns the mirror equation and does not predict the 2D mirror mask in novel views. When doing Novel View Synthesis，We directly use Ground Truth Mirror Region Mask**  

   However, in real applications, a more reasonable approach would be allowing the model to predict the 2D mirror mask. A possible improvement can be seen in methods like [Mirror-3DGS](https://arxiv.org/abs/2404.01168) and [MirrorGaussian](https://mirror-gaussian.github.io/), which introduce an additional 3D-GS property and loss to address this issue.  
   Therefore, when making comparisons, fairness should be carefully considered.  

2. The final rendering quality highly depends on the accuracy of the mirror equation, which is strongly influenced by the initialization quality. Due to the randomness of RANSAC, if the initialized mirror equation deviates significantly from the true value, the RGB loss in the camera pose optimization stage alone may not be sufficient to correct the error. In such cases, consider adjusting the RANSAC parameters, or improving camera pose optimization, e.g., by introducing a Feature Matching Loss to enhance robustness.  

## Citation  

If you find our work useful, please consider citing:  

```bibtex
@article{wang2024gaussian,
  title={Gaussian Splatting in Mirrors: Reflection-Aware Rendering via Virtual Camera Optimization},
  author={Wang, Zihan and Wang, Shuzhe and Turkulainen, Matias and Fang, Junyuan and Kannala, Juho},
  journal={arXiv preprint arXiv:2410.01614},
  year={2024}
}
```
