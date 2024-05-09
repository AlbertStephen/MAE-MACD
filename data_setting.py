import os
import torch
import torchvision
from PIL import Image
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset


def load_data_classifier(dataset, image_size= 224, batch_size = 64, shuffle= True, file_path = "./dataset/download/"):
    if dataset.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=file_path + "cifar10", train=True,transform= transform, download=True,),
                            batch_size=batch_size, shuffle=shuffle)
        # Testing Dataset
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=file_path + "cifar10", train=False,transform= transform,),
                            batch_size=batch_size, shuffle=shuffle)
    elif dataset.lower() == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=file_path + "cifar100", train=True, download=True,transform=transform),
                            batch_size=batch_size, shuffle=shuffle)

        # Testing Dataset
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=file_path + "cifar100", train=False,transform=transform),
                            batch_size=batch_size, shuffle=shuffle)
    elif dataset.lower() == "tinyimagenet":
        transform = transforms.Compose(
            [transforms.Resize(image_size),
             transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
             ])
        trainset = datasets.ImageFolder(root=os.path.join(file_path + 'tiny-imagenet-200', 'train'), transform=transform)
        testset = datasets.ImageFolder(root=os.path.join(file_path + 'tiny-imagenet-200', 'val'), transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    else:
        raise (f"{dataset} Error!")
    return train_loader, test_loader

def load_data_MAE(data, image_size, batch_size, data_path ="./dataset/download/"):
    # file_path =
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(image_size),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(0.5, 0.5),
         ])
    if data.lower() == "tinyimagenet":
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_path + 'tiny-imagenet-200', 'train'), transform=transform)
        test_set = torchvision.datasets.ImageFolder(root=os.path.join(data_path + 'tiny-imagenet-200', 'val'), transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    elif data.lower() == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(data_path + "cifar100", train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
        test_set = torchvision.datasets.CIFAR100(data_path + "cifar100", train=False, download=True, transform=transform)
    else:
        raise ("Data Name Wrong!")
    return train_loader, test_set


def Double_Dataloader(image_size= 224, batch_size= 64, data_name = "", data_path = "./dataset/"):
    # data_path = "./dataset/download/"
    if "cifar" in data_name.lower():
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_data_adv = Data_Loader(path=os.path.join(data_path, "generate/adv_" + data_name, "train/"),
                                     image_num=50000,
                                     batch_size=batch_size,
                                     suffix=".jpg",
                                     description="Train")
        test_data_adv = Data_Loader(path=os.path.join(data_path, "generate/adv_" + data_name, "test/"),
                                    image_num=10000,
                                    batch_size=batch_size,
                                    suffix=".jpg",
                                    description="Test",
                                    process=transform)
        if data_name.lower() == "cifar100":
            train_loader_ori = torchvision.datasets.CIFAR100(root=data_path + "download/cifar100",
                                                             train=True,
                                                             download=True)
            DoubleData = DoubleDataset(train_data_adv,
                                       train_loader_ori,
                                       transform1=transform,
                                       transform2=transform)
            Doubleloader = torch.utils.data.DataLoader(DoubleData,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            test_loader_adv = torch.utils.data.DataLoader(test_data_adv,
                                                          batch_size= batch_size,
                                                          shuffle= True)
            train_loader, test_loader = load_Data(dataset= "cifar100",
                                                  image_size=image_size,
                                                  batch_size=batch_size)

        elif data_name.lower() == "cifar10":
            train_loader_ori = torchvision.datasets.CIFAR10(root=data_path + "download/cifar10",
                                                            train=True,
                                                            download=True)
            DoubleData = DoubleDataset(train_data_adv,
                                       train_loader_ori,
                                       transform1=transform,
                                       transform2=transform)
            Doubleloader = torch.utils.data.DataLoader(DoubleData,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            test_loader_adv = torch.utils.data.DataLoader(test_data_adv,
                                                          batch_size=batch_size,
                                                          shuffle=True)
            train_loader, test_loader = load_Data(dataset="CIFAR-10",
                                                  image_size=image_size,
                                                  batch_size=batch_size)
        else: raise(f"{data_name} Error!" )

    elif data_name.lower() == "tinyimagenet":
        transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
            ])
        train_data_adv = Data_Loader(path=data_path + "generate/adv_tinyimagenet/train/",
                                     image_num=100000,
                                     batch_size=batch_size,
                                     suffix=".jpg",
                                     description="Train")
        test_data_adv = Data_Loader(path=data_path + "generate/adv_tinyimagenet/test/",
                                    image_num=10000,
                                    batch_size=batch_size,
                                    suffix=".jpg",
                                    description="Test",
                                    process=transform)
        train_data_ori = Data_Loader(path=data_path + "generate/tinyimagenet/train/",
                                     image_num=100000,
                                     batch_size=batch_size,
                                     suffix=".jpg",
                                     description="Train")
        test_data_ori = Data_Loader(path=data_path + "generate/tinyimagenet-200/test/",
                                    image_num=10000,
                                    batch_size=batch_size,
                                    suffix=".jpg",
                                    description="Test",
                                    process=transform)
        DoubleData = DoubleDataset_Tiny(train_data_adv,
                                        train_data_ori,
                                        transform1=transform,
                                        transform2=transform)
        Doubleloader = torch.utils.data.DataLoader(DoubleData, batch_size=batch_size, shuffle=True)

        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_path + "download/tiny-imagenet-200", 'train'),
                                                    transform=transform)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_path + 'download/tiny-imagenet-200', 'val'),
                                                   transform=transform)
        test_loader_adv = torch.utils.data.DataLoader(test_data_adv, batch_size=batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True)

    else:
        raise(f"{data_name} Error!" )
    return Doubleloader, test_loader_adv, train_loader, test_loader


class DoubleDataset(torch.utils.data.Dataset):
    def __init__(self, adv_data, ori_data, transform1=None, transform2= None):
        self.adv_data = adv_data.data
        self.adv_label = adv_data.targets
        self.ori_data = ori_data.data
        self.ori_label = ori_data.targets
        self.transforms_adv = transform1
        self.transforms_ori = transform2
    def __getitem__(self, index):
        adv_data = self.transforms_adv(Image.open(self.adv_data[index]))
        adv_label = self.adv_label[index]
        ori_data = self.transforms_ori(self.ori_data[index])
        ori_label = self.ori_label[index]
        return adv_data, adv_label, ori_data, ori_label
    def __len__(self):
        return len(self.adv_data)

class DoubleDataset_Tiny(torch.utils.data.Dataset):
    def __init__(self, adv_data, ori_data, transform1=None, transform2= None):
        self.adv_data, self.adv_label  = adv_data.data, adv_data.targets
        self.ori_data,self.ori_label  = ori_data.data, ori_data.targets
        self.transforms_adv, self.transforms_ori  = transform1, transform2

    def __getitem__(self, index):
        adv_data,adv_label = self.transforms_adv(Image.open(self.adv_data[index])), self.adv_label[index]
        ori_data, ori_label = self.transforms_ori(Image.open(self.ori_data[index])), self.ori_label[index]
        return adv_data, adv_label, ori_data, ori_label
    def __len__(self):
        return len(self.adv_data)


def Data_Loader(path, image_num, batch_size = 64, process = None, suffix = ".png", description = "Train"):
    assert os.path.exists(path), "path:%s may be wrong, Please Check!"%(path)
    image_file = []
    labels = []
    def generate_fold(path, image_number, suffix):
        image_sequence = []
        for i in range(image_number):
            image_sequence.append("IMAGE-" + str(i) + suffix)
        image_file_dir = []
        for i in range(len(image_sequence)):
            image_file_dir.append(path + image_sequence[i])
        label = np.loadtxt(path + 'label.txt', dtype=np.int)
        return image_file_dir, label
    image_file_dir, image_label = generate_fold(path = path, image_number = image_num, suffix= suffix)
    for image in image_file_dir:
        image_file.append(image)
    for label in image_label:
        labels.append(label)
    assert len(image_file) == len(labels), "the number between image and label is different, Please Check!"
    def Load(path):
        img_pil = Image.open(path)
        img_tensor = process(img_pil)
        return img_tensor
    data = dataset(image_file, labels, Load)
    return data

class dataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, label, loader):
        self.data = image_folder
        self.targets = label
        self.loader = loader
    def __getitem__(self, index):
        fn = self.data[index]
        img = self.loader(fn)
        target = self.targets[index]
        return img, target
    def __len__(self):
        return len(self.data)


def load_Data(dataset, image_size, batch_size, data_path = "./dataset/download/"):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if dataset.lower() == "cifar10":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=data_path + "cifar10", train=True, download=True,
                             transform=transforms), batch_size=batch_size, shuffle=True)
        # Testing Dataset
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=data_path + "cifar10", train=False,
                             transform=transforms), batch_size=batch_size, shuffle=True)
    elif dataset.lower() == "cifar100":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=data_path + "cifar100", train=True, download=True,
                              transform=transforms),
                              batch_size= batch_size, shuffle= True)
        # Testing Dataset
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=data_path + "cifar100", train=False,
                                          transform=transforms),
                                          batch_size=batch_size, shuffle=True)
    elif dataset.lower() == "tinyimagenet":
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(image_size),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
             ])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_path + 'tiny-imagenet-200', 'train'),
                                        transform=transform)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_path + 'tiny-imagenet-200', 'val'), transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    else:
         raise(f"{dataset} Error!")
    return train_loader, test_loader