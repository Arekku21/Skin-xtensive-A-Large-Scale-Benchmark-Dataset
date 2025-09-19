# Skin-xtensive-A-Large-Scale-Benchmark-Dataset

Official repository for the paper:

Skin-xtensive: A Large-Scale Benchmark Dataset for Deep Learning in Clinical Dermatology Towards Practical Deployment


## Overview

We introduce our main contributions:

1) Skin-xtensive dataset a comprehensive clinical image benchmarking dataset

2) Standardised benchmarking method


## Repository Contents
This repository contains:
- Benchmark model weights
- CSV file of categorical factor values

## Benchmark Model Weights

We provide the benchmark models trained weights on the Skin-xtensive dataset. Each model is available for download.

[Skin-xtensive Model Collection](https://huggingface.co/collections/Arekku21/skin-xtensive-a-large-scale-benchmark-dataset-68cd8f2a903a72b3af320a60)


| Model Type| Variant | M. Accuracy (%) | B. Accuracy (%) | Weights |
|------------|------------|------------|------------|------------|
| **CNN**      | ResNet152      | 54.90 ± 1.42 | 98.81 ± 0.51 |[Download](https://huggingface.co/Arekku21/skinxtensive-resnet152) |
|              | DenseNet121    | 57.11 ± 5.16 | 98.84 ± 0.38| [Download](https://huggingface.co/Arekku21/skinxtensive-densenet121)|
|              | InceptionV4    | 57.00 ± 5.97 | 99.08 ± 0.64 | [Download](https://huggingface.co/Arekku21/skinxtensive-inceptionv4)|
|              | EfficientNet-B5| 63.45 ± 5.08| 99.30 ± 0.34| [Download](https://huggingface.co/Arekku21/skinxtensive-efficientnetB5)|
|              | VGG19          | 55.60 ± 2.93| 99.08 ± 0.41 | [Download](https://huggingface.co/Arekku21/skinxtensive-vgg19)|
|              | MobileNetV3-L  | 54.90 ± 4.01 | 98.65 ± 0.38 | [Download](https://huggingface.co/Arekku21/skinxtensive-mobilenetv3)|
| **Transformer** | ViT-B/16   | 62.13 ± 3.75 | 99.51 ± 0.19 | [Download](https://huggingface.co/Arekku21/skinxtensive-vitb16)|
|              | **Swin-B/W7**| **70.70 ± 7.58** | **99.60 ± 0.23**| [Download](https://huggingface.co/Arekku21/skinxtensive-swinbw7)|
|              | CLIP (ViT-L/14)| 5.73 | 92.53| [Link](https://huggingface.co/openai/clip-vit-large-patch14)|

> **Note**: The reported accuracies and weights to download here are the best multiclass and binary results within our experiments. Please note *CLIP (ViT-L/14)* has no finetuned weight since it was just used directly for inference.


### Requirements

The downloadable model weights were taken from `Torch vision models`, `Timm` or `Huggingface`. This project was tested on `Python 3.9.21`. We have also included the `requirements.txt` for the python packages versions. 
 
Below is the preview of the dependencies (see [requirements.txt](requirements.txt) for the full list):

```txt
timm==1.0.7
torch==2.0.1
torchaudio==2.0.2
torchvision==0.15.2
transformers==4.47.1

```
## Skin-xtensive dataset categorical factors

We have provided the CSV file with the categorical factor values we have used for the performance analysis. Below is a snippet from the CSV file (see [test_normal_split_with_fitz_sharpness_exposure.csv](test_normal_split_with_fitz_sharpness_exposure.csv) for the full list):

| image_name| label | final_confirmed_diagnosis | fitzpatrick_label| exposure| sharpness|
|------------|------------|------------|------------|------------|------------|
|hyperpigmentation8.jpg| 27 | drug induced pigmentary changes | 4 |172.74018072289158| 8478.191137|
|IMG1807.jpg| 1 | Lepromatous leprosy | 6 |145.10880533854166| 72.41024805466334|
|img_imageId=6012.jpg| 64 | porokeratosis of mibelli | 4 |162.29630528824933| 1475.5468036193238|

**Column descriptions**
 - `image_name` - File name of the image in the dataset
 - `label` - Numberic class ID 
 - `final_confirmed_diagnosos` -  Confirmed dermatoligical diagnosis 
 - `fitzpatrick_label` -  Fitzpatrick skin type (1-6)
 - `exposure` -  image exposure score (how bright)
 - `sharpness` - image sharpness score (how sharp)


 ## Citation

 If you use this repository or the Skin-xtensive models in your research, please cite our work: 
 
> **Note**: Our paper is currently in review and this citation will change

```bibtex
@misc{skinxtensive2025,
    author       = {Arekku21 and contributors},
    title        = {Skin-xtensive: A Large-Scale Benchmark Dataset for Deep Learning in Clinical Dermatology Towards Practical Deployment},
    year         = {2025},
    publisher    = {GitHub},
    journal      = {GitHub repository},
    howpublished = {\url{https://github.com/Arekku21/Skin-xtensive-A-Large-Scale-Benchmark-Dataset}},
}
```


