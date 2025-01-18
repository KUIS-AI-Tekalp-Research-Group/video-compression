# Motion-Adaptive Inference for Flexible Learned B-Frame Compression

This repository contains the official implementation of the paper:

**Motion-Adaptive Inference for Flexible Learned B-Frame Compression**  
by M. Akin Yilmaz, O. Ugur Ulas, Ahmet Bilican, and A. Murat Tekalp

[arXiv preprint](https://arxiv.org/abs/2402.08550)

## Abstract
While the performance of recent learned intra and sequential video compression models exceed that of respective traditional codecs, the performance of learned B-frame compression models generally lag behind traditional B-frame coding. The performance gap is bigger for complex scenes with
large motions. This is related to the fact that the distance between the past and future references vary in hierarchical Bframe compression depending on the level of hierarchy, which
causes motion range to vary. The inability of a single Bframe compression model to adapt to various motion ranges
causes loss of performance. As a remedy, we propose controlling the motion range for flow prediction during inference
(to approximately match the range of motions in the training
data) by downsampling video frames adaptively according to
amount of motion and level of hierarchy in order to compress
all B-frames using a single flexible-rate model. We present
state-of-the-art BD rate results to demonstrate the superiority
of our proposed single-model motion-adaptive inference approach to all existing learned B-frame compression models

## Pretrained Model Weights

You can download the pretrained checkpoint for our model from the following link:

[Download Pretrained Checkpoint](https://drive.google.com/drive/folders/1Q7YIJeWeSYyFGYcYN5xlPKr22YsBz4Fi?usp=drive_link)

You can also download the pretrained Intra codec (ELIC) from the following link:

[Download Pretrained ELIC](https://drive.google.com/drive/folders/1WGAVUQaL6wzBqQkjXko0ToMV4QHijVpK?usp=drive_link)

## Requirements



## Testing
```bash
python main.py
```

## Citation
If you use this code, please cite our paper:

```bibtex
@misc{yilmaz2024motionadaptiveinferenceflexiblelearned,
      title={Motion-Adaptive Inference for Flexible Learned B-Frame Compression}, 
      author={M. Akin Yilmaz and O. Ugur Ulas and Ahmet Bilican and A. Murat Tekalp},
      year={2024},
      eprint={2402.08550},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2402.08550}
}
