# Detecting fake images

**Towards Universal Fake Image Detectors that Generalize Across Generative Models** <br>
[Utkarsh Ojha*](https://utkarshojha.github.io/), [Yuheng Li*](https://yuheng-li.github.io/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) <br>
(*Equal contribution)


[[Project Page](https://utkarshojha.github.io/universal-fake-detection/)] [[Paper](https://arxiv.org/abs/2302.10174)]

<p align="center">
    <a href="https://utkarshojha.github.io/universal-fake-detection/"><img src="resources/teaser.png" width="50%">></a> <br>
    Using images from one breed of generative model (e.g., GAN), detect fake images from other **breeds** of generative models (e.g., Diffusion models)
</p>

## Contents

- [Setup](#setup)
- [Pretrained model](#weights)
- [Data](#data)
- [Evaluation](#evaluation)
- [Training](#training)


## Setup 

1. Clone this repository 
```bash
git clone https://github.com/Yuheng-Li/UniversalFakeDetect
cd UniversalFakeDetect
```

2. Install the necessary libraries
```bash
pip install torch torchvision
```

## Pretrained model

- The pretrained weights for the linear probing model can be found here: [link]()
- Download and save the model in the `checkpoints` folder.


## Data

- Of the 19 models studied overall (Table 1/2 in the main paper), 11 are taken from a [previous work](https://arxiv.org/abs/1912.11035). Download the test set, i.e., real/fake images for those 11 models given by the authors from [here](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view) (dataset size ~19GB).
- Download the file and unzip it in `datasets/test`.
- This should create a directory structure as follows:
```

datasets
â””â”€â”€ test			
      â””â”€â”€ c0			
	   â”œâ”€â”€ progan	
	   â”‚â”€â”€ cyclegan   	
	   â”‚â”€â”€ biggan
	   â”‚      .
	   â”‚      .
                  
```
- Each directory (e.g., progan) will contain real/fake images under `0_real` and `1_fake` folders respectively.
- Dataset for the diffusion models (e.g., LDM/Glide) used in the paper will be released soon.


## Evaluation
 
- You can evaluate the model on all the dataset at once by running:
```bash
python validate.py  --arch=CLIP:ViT-L/14   --ckpt=checkpoints/model.pth   --result_folder=clip_vitl14 
```

- You can also evaluate the model on one generative model by specifying the paths of real and fake datasets
```bash
python validate.py  --arch=CLIP:ViT-L/14   --ckpt=checkpoints/model.pth   --result_folder=clip_vitl14  --real_path datasets/test/progan/0_real --fake_path datasets/test/progan/1_fake
```

Note that if no arguments are provided for `real_path` and `fake_path`, the script will perform the evaluation on all the domains specified in `dataset_paths.py`.

- The results will be stored in `results/<folder_name>` in two files: `ap.txt` stores the Average Prevision for each of the test domains, and `acc.txt` stores the accuracy (with 0.5 as the threshold) for the same domains.

## Training

- Our main model is trained on the same dataset used by the authors of [this work](https://arxiv.org/abs/1912.11035). Download the official training dataset provided [here](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view) (dataset size ~ 72GB). 

- Download and unzip the dataset in `datasets/train` directory. The overall structure should look like the following:
```
datasets
â””â”€â”€ train			
      â””â”€â”€ progan			
           â”œâ”€â”€ airplane
           â”‚â”€â”€ bird
           â”‚â”€â”€ boat
           â”‚      .
           â”‚      .
```
- A total of 20 different object categories, with each folder containing the corresponding real and fake images in `0_real` and `1_fake` folders.
	
## Citation

If you find our work helpful in your research, please cite it using the following:
```bibtex
@inproceedings{ojha2023fakedetect,
      title={Towards Universal Fake Image Detectors that Generalize Across Generative Models}, 
      author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
      booktitle={CVPR},
      year={2023},
}
```
