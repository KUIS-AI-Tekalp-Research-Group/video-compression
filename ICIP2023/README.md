# Motion-Adaptive Inference for Flexible Learned B-Frame Compression

This repository contains the official implementation of the paper:

**Multi-scale Deformable Alignment and Content-adaptive Inference for Flexible-rate Bi-directional Video Compression**  
by M. Akin Yilmaz, O. Ugur Ulas, and A. Murat Tekalp

[arXiv preprint](https://arxiv.org/abs/2306.16544)

## Abstract
The lack of ability to adapt the motion compensation model to video content is an important limitation of current endto-end learned video compression models. This paper advances the state-of-the-art by proposing an adaptive motioncompensation model for end-to-end rate-distortion optimized hierarchical bi-directional video compression. In particular, we propose two novelties: i) a multi-scale deformable alignment scheme at the feature level combined with multi-scale conditional coding, ii) motion-content adaptive inference. In addition, we employ a gain unit, which enables a single model to operate at multiple rate-distortion operating points. We also exploit the gain unit to control bit allocation among intra-coded vs. bi-directionally coded frames by fine tuning corresponding models for truly flexible-rate learned video coding. Experimental results demonstrate state-of-the-art rate-distortion performance exceeding those of all prior art in learned video coding.

## Pretrained Model Weights

You can download the pretrained weights for our model from the following link:

!! Coming in a couple of days


## Testing
```bash
python main.py
```

## Citation
If you use this code, please cite our paper:

```bibtex
@INPROCEEDINGS{multiscale_deform_bidirec,
  author={Yılmaz, M. Akın and Ugur Ulas, O. and Tekalp, A. Murat},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)}, 
  title={Multi-Scale Deformable Alignment and Content-Adaptive Inference for Flexible-Rate Bi-Directional Video Compression}, 
  year={2023},
  volume={},
  number={},
  pages={2475-2479},
  keywords={Deformable models;Video coding;Adaptation models;Image coding;Bit rate;Rate-distortion;Bidirectional control;bi-directional video compression;hierarchical B pictures;end-to-end rate-distortion optimization;content-adaptive inference;flexible-rate coding},
  doi={10.1109/ICIP49359.2023.10223112}}

