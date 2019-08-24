import torch.nn as nn
from torchvision.models import ResNet, resnet34, resnext50_32x4d
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
from spacecutter.models import OrdinalLogisticModel

class resnet34_pretrained(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(resnet34_pretrained, self).__init__()
        self.resnet = resnet34(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

class resnext50_pretrained(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(resnext50_pretrained, self).__init__()
        self.model = resnext50_32x4d(pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class efficientnet_pretrained(nn.Module):
    def __init__(self, k, num_classes, pretrained):
        super(efficientnet_pretrained, self).__init__()
        self.name = 'efficientnet-b' + str(k)
        self.model = EfficientNet.from_pretrained(self.name, num_classes)
    
    def forward(self, x):
        return self.model(x)


class cadene_model(nn.Module):
    def __init__(self, model_name, num_classes):
        super(cadene_model, self).__init__()
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        num_features = self.model.last_linear.in_features
        self.model.last_linear = nn.Linear(num_features, num_classes)
        self.model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        return self.model(x)


class ordinal_efficientnet(nn.Module):
    def __init__(self, k, num_classes, pretrained):
        super(ordinal_efficientnet, self).__init__()
        self.name = 'efficientnet-b' + str(k)
        model_reg = EfficientNet.from_pretrained(self.name, 1)
        self.model = OrdinalLogisticModel(model_reg, num_classes)

    def forward(self, x):
        cutpoints = self.model.link.cutpoints.data
        for i in range(cutpoints.shape[0] - 1):
            cutpoints[i].clamp_(-2, cutpoints[i + 1])
        return self.model(x)    

