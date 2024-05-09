import random
import torch
import numpy as np
import argparse
from einops.layers.torch import Rearrange
from PIL import Image
from torchattacks import PGD, FGSM, Jitter, AutoAttack
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


# # Generate a mask matrix to occulude the image of the classifier
def mask_image(image, patch_size, masking_ratio, device):
    assert 0 <= masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
    def pair(t):
        return t if isinstance(t, tuple) else (t, t)
    batch_size, channels, image_height, image_width = image.size()
    patch_height, patch_width = pair(patch_size)
    assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
    # # Compute the total number of image patch
    num_patches = (image_height // patch_height) * (image_width // patch_width)
    # # Compute the number of mask patches
    num_masked = int(masking_ratio * num_patches)
    # # Generate random mask and unmasked patches index
    rand_indices = torch.rand(batch_size, num_patches, device= device).argsort(dim=-1)
    masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
    # # mask patches matrix
    mask_matrix = torch.zeros((batch_size, num_patches, patch_height, patch_width))
    # # set unmask matrix value into 1, in terms of unmasked indexes
    batch_range = torch.arange(batch_size)[:, None]
    mask_matrix[batch_range, unmasked_indices] = 1
    # # transform mask patches matrix to mask image matrix
    raarrange = Rearrange('b (h w) p1 p2-> b (h p1) (w p2)', h=image_height // patch_height, w=image_height // patch_height)
    mask_matrix = raarrange(mask_matrix)
    mask_matrix = mask_matrix.unsqueeze(1)
    # # Generate mask image
    mask_matrix = mask_matrix.to(image.device)
    image = image * mask_matrix
    return image


# Function Name: attack
# Input: loader: this is a torch dataloader
#        model: this a torch NN model with parameters
# function: This program generates a adversarial images and correspond labels,
#           according to model parameters, dataset
# Usage: adv_images, adv_lables = attack(dataloader)
def attack(model_def, model_atk, loader, attack_index, eps, save_path = None):
    correct = 0
    total_adv = 0
    model_atk.to(device)
    model_def.to(device)
    label = np.array(loader.dataset.targets)
    if save_path is not None:
        np.savetxt(save_path + "label.txt", label, fmt = '%d')
    atk = atk_method(attack_index, model_atk, eps)
    for index, (images, labels) in enumerate(tqdm(loader)):
        torch.cuda.empty_cache()
        images = images.to(device)
        adv_images = atk(images, labels)
        if save_path is not None:
            # # CIFAR10/CIFAR100
            # save_advimage(image= adv_images, index= index, std= [0.229, 0.224, 0.225], mean= [0.485, 0.456, 0.406], img_size= 32,save_path= save_path)
            # # TINY IMAGENET
            save_advimage(image=adv_images, index=index, std=[0.4802, 0.4481, 0.3975], mean=[0.2770, 0.2691, 0.2821], img_size=64, save_path=save_path)
        outputs = model_def(adv_images).to(device)
        pred = outputs.max(1, keepdim=True)[1].cpu()
        compare = pred.eq(labels.view_as(pred))
        correct += compare.sum().item()
        total_adv += labels.size(0)
    accuracy = (100 * float(correct) / total_adv)
    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total_adv))
    return accuracy


# Function Name: save_advimage
# Input: image(B, C, W, H): this is a torch tensor
#        index(int): denotes the index of image in the dataloader
#        mean, std(list(float, float, float)): Normalize parameters
#        save_path(str):target directory
#        img_size: the save size of adversarial images
# function: according corresponding parameters save images
# Usage: save_advimage(image, index, mean, std, img_size, save_path ="./dataset/CIFAR100/train/")
def save_advimage(image, index, mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225], img_size= 32, save_path ="./dataset/CIFAR100/train/"):
    std_tensor = torch.tensor(std, device=image.device).view(1, -1, 1, 1)
    mean_tensor = torch.tensor(mean, device=image.device).view(1, -1, 1, 1)
    image = (image * std_tensor + mean_tensor) * 255
    for i in range(image.size(0)):
        img = image[i].to("cpu").numpy().astype(np.uint8)
        img1 = (Image.fromarray(img.transpose(1, 2, 0))).resize([img_size, img_size])
        name_str = save_path + "IMAGE-{}.jpg".format(index * 64 + i)
        img1.save(name_str)

# Load attack method
def atk_method(index, model, eps):
    if index == 0:
        return FGSM(model, eps= eps)
    elif index == 1:
        return PGD(model, eps= eps, alpha=8/255, steps=40)
    elif index == 2:
        return AutoAttack(model, norm='L2', eps= eps)
    elif index == 3:
        return Jitter(model, eps=eps, alpha=2/255, steps=40)
    else:
        raise("index Error")


class ntxent_loss(torch.nn.Module):
    def __init__(self, tau=1):
        super(ntxent_loss, self).__init__()
        self.tau = tau
        self.sim = torch.nn.CosineSimilarity(dim= 2)

    def forward(self, xi, xj, criterion = torch.nn.CrossEntropyLoss(), device=torch.device("cuda")):
        batch_size = xi.size(0)
        LABELS = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float()  # one-hot representations
        LABELS = LABELS.to(device)
        features = torch.cat((xi, xj), dim=0)
        features = torch.nn.functional.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(LABELS.shape[0], dtype=torch.bool).to(device)
        labels = LABELS[~mask].view(LABELS.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        logits = logits / self.tau
        return criterion(logits, labels)

class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = torch.nn.functional.log_softmax(y_s/self.T, dim=1)
        p_t = torch.nn.functional.softmax(y_t/self.T, dim=1)
        loss = torch.nn.functional.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True