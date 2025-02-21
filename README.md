# ICME2024 - Multi-batch Nuclear-norm Adversarial Network for Unsupervised Domain Adaptation

[![Paper](https://img.shields.io/badge/Paper-official-green)](https://ieeexplore.ieee.org/abstract/document/10687425) [![Paper](https://img.shields.io/badge/PDF-Poster-blue)](https://github.com/YuZhenyuLindy/Yuan/blob/main/MBAN_ICME_2024_Poster_00.png) [![PPT](https://img.shields.io/badge/PDF-Slides-orange)](https://github.com/YuZhenyuLindy/Yuan/blob/main/MBAN_ICME_2024.pdf) [![YouTube](https://img.shields.io/badge/Video-YouTube-red)](https://youtu.be/AnSf7OFx71M)

# MBAN 

This repo is the official PyTorch implementation of "Multi-batch Nuclear-norm Adversarial Network for Unsupervised Domain Adaptation".

<div align=center><img src="./MBAN_ICME_2024_Poster_00.png" width="100%"></div>

## Abstract

Adversarial learning has achieved great success for unsupervised domain adaptation (UDA). Existing adversarial UDA methods leverage the predicted discriminative information with Nuclear-norm Wasserstein discrepancy for feature alignment. However, the limited memory space makes it very difficult to accurately calculate the Nuclear-norm, which hinders domain adaptation. To address this challenge, we propose the multi-batch Nuclear-norm adversarial network, termed as MBAN. Specifically, we build a dynamic queue to cache feature, which encourages to generate a large and consistent output matrix, enabling accurately calculating the Nuclear-norm. Then, the multi-batch Nuclear-norm discrepancy is proposed, which can effectively improve the transferability and discriminability of the learned features. Experimental results show that MBAN could achieve significant performance improvement, especially when the number of categories is quite large. 


## Prepare
```bash
pip install -r requirements.txt
```


## Datasets
* [DomainNet](http://ai.bu.edu/M3SDA/)
* [Office31](https://faculty.cc.gatech.edu/~judy/domainadapt/)
* [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)
* [VisDA2017](http://ai.bu.edu/visda-2017/)

## Contact
For any questions, feel free to reach out:

📧 Email: Pei Wang (peiwang0518@163.com), Zhenyu Yu (yuzhenyuyxl@foxmail.com)

## Bibtex
If you find this work useful for your research, please cite:
```bibtex
@inproceedings{yu2024capan,
  title={CaPAN: Class-aware Prototypical Adversarial Networks for Unsupervised Domain Adaptation},
  author={Yu, Zhenyu and Wang, Pei},
  booktitle={2024 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```


## Acknowledgment
Some codes are mainly based on following repositories. Thanks for their contribution.
* [DALN](https://github.com/xiaoachen98/DALN.git)
* [SCDA](https://github.com/BIT-DA/SCDA.git)
* [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library.git)
