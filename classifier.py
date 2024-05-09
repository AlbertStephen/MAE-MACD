import torch, torchvision
from torchvision import models
from tqdm import tqdm, trange
# from My_Dataloader import load_data_classifier

save_name = ["_ResNet50.pth", "_WideResNet50.pth"]
def Resnet50(num_classes= 100, pretrained= False):
    model = models.resnet50(pretrained=pretrained)
    model.fc = torch.nn.Linear(in_features= 2048, out_features= num_classes, bias= True)
    model = SplitModel(model)
    return model

def WideResNet50(num_classes, pretrained= True):
    model = torchvision.models.wide_resnet50_2(pretrained= pretrained)
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    model = SplitModel(model)
    return model

class SplitModel(torch.nn.Module):
    def __init__(self, model):
        super(SplitModel, self).__init__()
        self.backbone, self.classifier = Split(model)

    def forward(self, x):
        output_backbone = self.backbone(x)
        output_backbone = torch.flatten(output_backbone, 1)
        output_classifier = self.classifier(output_backbone)
        return output_classifier

def Split(model):
    backbone = torch.nn.Sequential()
    classifier = torch.nn.Sequential()
    num_child = len(list(model.named_children()))
    for index, (i) in enumerate(model.named_children()):
        if index != num_child - 1:
            backbone.add_module(i[0], i[1])
        else:
            classifier = i[1]
    return backbone, classifier

def load_model(dataset= "cifar100", index = 0, pretrained = True):
    if dataset.lower() == "cifar10":
        num_classes = 10
    elif dataset.lower() == "cifar100":
        num_classes = 100
    elif dataset.lower() == "tinyimagenet":
        num_classes = 200
    else:
        raise(f"{dataset} Error!")

    if index == 0:
        model = Resnet50(num_classes= num_classes, pretrained= pretrained)
    elif index == 1:
        model = WideResNet50(num_classes= num_classes, pretrained= pretrained)
    else:
        raise{f"{index} Error!"}
    return model

def load_ModelParam(dataset= "cifar100", index = 0, file_path = "./"):
    model = load_model(dataset, index)
    model_param = torch.load(file_path + dataset.upper() + save_name[index])
    model.backbone.load_state_dict(model_param["backbone"])
    model.classifier.load_state_dict(model_param["classifier"])
    return model

def train_model(model, epochs, train_loader, file_name, criteria = torch.nn.CrossEntropyLoss(), test_loader=None, device = "cuda", is_save = False):
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0005)
    for _ in trange(epochs, desc="{:>12}".format("Train Model")):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data).to(device)
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()
    if is_save:
        backbone = model.backbone
        classifier = model.classifier
        save_param = {"backbone": backbone.state_dict(), "classifier": classifier.state_dict()}
        torch.save(save_param, file_name)


@torch.no_grad()
def test(model, dataloader, description= None, device = "cuda"):
    model.eval()
    test_loss = 0
    correct = 0
    label = []
    model.to(device)
    if description == None:
        description = "Test Model"
    for index, (data, target) in enumerate(tqdm(dataloader, desc="{:>12}".format(description))):
        data, target = data.to(device), target.to(device)
        output = model(data).to(device)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        label += list((pred.cpu().numpy())[:,0])
    test_loss /= len(dataloader.dataset)
    print('Accuracy: {}/{} ({:.0f}%)'
          .format(correct, len(dataloader.dataset),
                  100. * correct / len(dataloader.dataset)))
    return correct / len(dataloader.dataset)

if __name__ == "__main__":
    datasets = ["cifar10", "cifar100", "tinyimagenet"]
    indexes = [0, 1]
    for d in datasets:
        train_loader, test_loader = load_dataloader(dataset=d)
        for i in indexes:
            model = load_ModelParam(dataset=d, index=i, file_path="./")
            test(model, test_loader, description=None, device="cuda")
