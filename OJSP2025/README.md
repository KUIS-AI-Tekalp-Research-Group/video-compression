# Content-Adaptive Inference for State-of-the-Art Learned Video Compression

This repository contains the reference implementation of the **Content-Adaptive Inference** framework described in:

> **Bilican, A., Yilmaz, M. A., & Tekalp, A. M.** (2025). *Content-Adaptive Inference for State-of-the-Art Learned Video Compression*. IEEE Open Journal of Signal Processing, 6, 498–506. doi:10.1109/OJSP.2025.3564817

The python file contains an example implementation of the proposed method. One can easilty adapt this code to needed repository. 
In the [DCVC-FM](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-FM) repository under the src/models directory the base implementation of python file above can be found. 

## Features

- **Adaptive Frame Downsampling**  
  Selects a per-frame downsampling factor to match test-time motion ranges to those seen during training, improving flow estimation and prediction accuracy.
- **Motion Vector Scaling**  
  Scales the estimated flow field at inference to keep motion statistics within the compressor’s training distribution, reducing rate without sacrificing PSNR.
- **Model-Agnostic Encoder/Decoder Hooks**  
  Minimal changes to the standard DCVC-FM encoder/decoder—no model retraining or fine-tuning required.

```
@article{Bilican2025ContentAdaptive,
  title   = {Content-Adaptive Inference for State-of-the-Art Learned Video Compression},
  author  = {Bilican, Ahmet and Yilmaz, M. Akin and Tekalp, A. Murat},
  journal = {IEEE Open Journal of Signal Processing},
  volume  = {6},
  pages   = {498--506},
  year    = {2025},
  doi     = {10.1109/OJSP.2025.3564817}
}
```
