import torch
import argparse
import math
from einops.layers.torch import Rearrange
import numpy as np
from IMAE import Teacher_MAE, take_indexes
from tqdm import tqdm, trange
from data_setting import load_data_MAE,  Double_Dataloader
from utils import mask_image, ntxent_loss, DistillKL, attack, add_dict_to_argparser
from classifier import load_ModelParam, test, save_name



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))


def train_MAE(model, epoches):
    optim = torch.optim.AdamW(model.parameters(), lr= 1.5e-4 * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=0.05)
    lr_func = lambda epoch: min((epoch + 1) / (200 + 1e-8),
                                0.5 * (math.cos(epoch / epoches * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)
    step_count = 0
    feature_criterion = torch.nn.MSELoss()
    optim.zero_grad()
    for e in range(epoches):
        model.train()
        losses = []
        for img, label in tqdm(iter(train_loader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask, feature_patch, feature_mask = model(img)
            loss_img = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            # # Squeeze Information
            loss_feature = feature_criterion(feature_mask[:, 0], feature_patch[:, 0])
            loss = loss_img + loss_feature
            loss.backward()
            if step_count % 1 == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')


# @torch.no_grad()
def defence_test(model_def, data, classifier_index, file_path, attack_index, epsilon):
    model_atk = load_ModelParam(dataset= data,
                                index= classifier_index,
                                file_path = file_path)
    test(model_def, test_loader, device="cuda")
    attack(model_atk=model_atk,
           model_def=model_def,
           loader=test_loader,
           attack_index=attack_index,
           eps=epsilon)

class Teacher(torch.nn.Module):
    def __init__(self, model, process):
        super(Teacher, self).__init__()
        self.encoder = model
        self.process = process
    @torch.no_grad()
    def forward(self, img):
        patches, mask_patches, forward_indexes, backward_indexes = self.process(img)
        features = self.encoder(patches)
        mask = torch.zeros_like(patches)
        mask[mask_patches.size(0):] = 1
        patch_size = 16
        image_size = 224
        rearrange = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        mask = rearrange(mask)
        return features, mask

def MACD(teacher, student, epoches, patch_size):
    classifier_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.01)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0005)
    student.to(device)
    student.train()
    teacher.eval()
    Loss_Distill = DistillKL(T=4)
    loss_contrast = ntxent_loss()
    masking_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    for i in trange(epoches, desc="{:>10}".format("feature learning")):
        for index, (adv_image, target, ori_image, target1) in enumerate(Doubleloader):
            input1, input2 = adv_image, ori_image
            output_teacher1, mask1 = teacher(input1.to(device))
            # Random Masking ratio
            mask_ratio = masking_list[np.random.randint(49) % 5]
            input1 = mask_image(image=adv_image,
                                patch_size=patch_size,
                                masking_ratio=mask_ratio,
                                device= device)
            input2 = mask_image(image=ori_image,
                                patch_size=patch_size,
                                masking_ratio=mask_ratio,
                                device= device)
            # # adversarial and benign examples is input into student.backbone
            output_student1 = student.backbone(input1.to(device)).squeeze()
            output_student2 = student.backbone(input2.to(device)).squeeze()
            # # output current label
            output = student.classifier(output_student2.flatten(1))
            # # Compute loss
            loss = (Loss_Distill(output_student2, output_teacher1[:, 0]) +
                    loss_contrast(output_student1, output_student2) +
                    classifier_loss(output, target.to(device)))
            # # backward optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def create_argparser(dataset= "cifar10", classifier_index= 0):
    defaults = dict(
        dataset=dataset,
        mask_ratio=0.85,
        batch_size=32,
        image_size=224,
        MAE_epoches=1,
        MAE_path="./model/mae/CIFAR100-IMAE.pt",
        classifier_index=classifier_index,
        classifier_epoches = 1,
        classifier_path = "./model/classifier/",
        robust_classifier_path = "./model/robust_classifier/",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser = parser.parse_args()
    return parser


if __name__ == '__main__':
    # # datasets = ["cifar10", "cifar100", "tinyimagenet"]
    # # classifier_index = 0, 1; 0 denotes ResNet50; 1 denotes WideResNet50;
    MAE = Teacher_MAE(mask_ratio=0.75).to(device)
    args = create_argparser(dataset = "cifar100")
    print(args)
    train_loader, val_dataset = load_data_MAE(args.dataset,
                                              image_size=args.image_size,
                                              batch_size=args.batch_size)
    train_MAE(MAE, args.MAE_epoches)
    # torch.save(MAE, args.MAE_save_path)
    Doubleloader, test_loader_adv, train_loader, test_loader = Double_Dataloader(data_name=args.dataset,
                                                                                 image_size=args.image_size,
                                                                                 batch_size=args.batch_size)
    classifier = load_ModelParam(dataset=args.dataset,
                                 index = args.classifier_index,
                                 file_path=args.classifier_path)
    # MAE = torch.load(args.MAE_path).to("cpu")
    # # Extract Teacher Model
    teacher = Teacher(MAE.new_encoder, MAE.image_patch).to(device)
    # # Train MACD
    MACD(teacher, classifier, args.classifier_epoches, patch_size=8)
    torch.save(classifier.state_dict(),
               args.classifier_save_path + (args.dataset).upper() + save_name[args.classifier_index])
    defence_test(model_def=classifier,
                 data=args.dataset,
                 classifier_index=args.classifier_index,
                 file_path=args.classifier_path,
                 attack_index=1,
                 epsilon=8 / 255)






