# MAE-MACD
This project is the code implementation of the paper "MAE-MACD: The Masked Adversarial Contrastive Distillation Algorithm Grounded in Masked Auto-Encoders".

# Project Description

This project is the code implementation of the paper "MAE-MACD: The Masked Adversarial Contrastive Distillation Algorithm Grounded in Masked Auto-Encoders".

## File and Directory Structure

- `dataset`: Contains the image dataset used in the paper, primarily involving downloading and locally generating image files.

- `model`: Includes the parameter of models used in the paper, including standard classifiers, robust classifiers, and the improved MAE. 

- `classifier.py`: Implements standard classifiers ResNet50 and WideResNet50, along with corresponding training and testing routines.

- `TMAE.py`: Implements the standard Masked Auto-Encoder, sourced from the [GitHub project](https://github.com/IcarusWizard/MAE).

- `IMAE.py`: Implements the improved Masked Auto-Encoder proposed in this paper.

- `utils.py`: Contains various utility functions used in the algorithm implementation, including masking algorithm `mask_image`, adversarial attack `attack`, contrastive loss `ntxent_loss`, distillation loss `DistillKL`, and training parameter updates `add_dict_to_argparser`, among others.

- `data_setting.py`: Implements the dataloader format used in the paper, including standard training set, IMAE training set, and MACD training set.

- `main.py`: Contains the training code for IMAE and MACD, as well as the adversarial testing code for MACD.

# # Model Files Location

The model files for the standard classifier, robust classifier, and IMAE, as well as the adversarial images used in this paper, are located on [Google Drive](https://drive.google.com/drive/folders/16tFEZHbSHCLvkBr4xEgdawvHrvxXR6qF?usp=drive_link). Please note that the adversarial images need to be unzipped.
