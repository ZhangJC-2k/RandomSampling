<div align="center">
  
# Random Sampling for Diffusion-based Adversarial Purification [![arXiv](https://img.shields.io/badge/Arxiv-Paper-179bd3)](https://arxiv.org/abs/2411.18956)

Jiancheng Zhang, Peiran Dong, Yongyong Chen, Yin-Ping Zhao, Song Guo
</div>

This is a repo for our work: "**[Random Sampling for Diffusion-based Adversarial Purification](https://arxiv.org/abs/2411.18956)**".

### News

**2024.11.28 :** Our paper is available on Arvix and the code is posted here.

## Abstract

Denoising Diffusion Probabilistic Models (DDPMs) have gained great attention in adversarial purification. Current diffusion-based works focus on designing effective condition-guided mechanisms while ignoring a fundamental problem, i.e., the original DDPM sampling is intended for stable generation, which may not be the optimal solution for adversarial purification. Inspired by the stability of the Denoising Diffusion Implicit Model (DDIM), we propose an opposite sampling scheme called random sampling. In brief, random sampling will sample from a random noisy space during each diffusion process, while DDPM and DDIM sampling will continuously sample from the adjacent or original noisy space. Thus, random sampling obtains more randomness and achieves stronger robustness against adversarial attacks.
Correspondingly, we also introduce a novel mediator conditional guidance to guarantee the consistency of the prediction under the purified image and clean image input. To expand awareness of guided diffusion purification, we conduct a detailed evaluation with different sampling methods and our random sampling achieves an impressive improvement in multiple settings. Leveraging mediator-guided random sampling, we also establish a baseline method named DiffAP, which significantly outperforms state-of-the-art (SOTA) approaches in performance and defensive stability. Remarkably, under strong attack, our DiffAP even achieves a more than 20% robustness advantage with $10\times$ sampling acceleration. 

## Differences in sampling trajectories
<img src="./Figure/sampling_trajectory.png">
Conceptual illustration of the sampling trajectory of three different diffusion processes. The orange center area represents the distribution of clean samples, and the outer circle represents the sampling space after the noise disturbance. The noise intensity on the same circle is the same when the noise disturbance on the outer circle is greater.

## 1. Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- [PyTorch >= 1.8](https://pytorch.org/)

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  pip install -r requirements.txt
  ```

## 2.Prepare Pretrained models and data
Before running our code, download the necessary pre-trained models and dataset. For CIFAR-10, we use the diffusion model identical to those in [DiffPure](https://github.com/NVlabs/DiffPure#requirements), which is available in [ScoreSDE](https://github.com/yang-song/score_sde_pytorch). We also use pre-trained classifiers, and for CIFAR-10, you don't need to download the models. After downloading the models and dataset, you must update their path in 'path.py'.

## 3.Test
There are several arguments to configure both attacks and defenses. 
| Arguments | Explanation | 
|:-|:-|
| `attack_method` | Specific type of attack |
| `n_iter` | The number of update iterations of the attack |
| `eot` | The number of EOT (Expectation Over Transformation) samples |
| `def_max_timesteps` | The number of forward diffusion steps of the defend |
| `def_num_denoising_steps` | The number of reverse sampling steps of the defend |
| `def_sampling_method` | The sampling method of the defend |
| `def_guidance` | The condition guidance of the defend |
| `att_max_timesteps` | The number of forward diffusion steps of the attack |
| `att_num_denoising_steps` | The number of reverse sampling steps of the attack |
| `att_sampling_method` | The sampling method of the attack |
| `att_guidance` | The condition guidance of the attack |

After configuring the arguments, run the following commands:
```shell
cd RandomSampling 
python test.py
```

## Cite
If this repo helps you, please consider citing our work:
```
@article{zhang2024random,
  title={Random Sampling for Diffusion-based Adversarial Purification},
  author={Zhang, Jiancheng and Dong, Peiran and Chen, Yongyong and Zhao, Yin-Ping and Guo, Song},
  journal={arXiv preprint arXiv:2411.18956},
  year={2024}
}
```
