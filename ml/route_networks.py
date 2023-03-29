import torch
import torch.nn as nn
import torchvision

# Inspired by https://github.com/pytorch/examples/blob/main/siamese_network/main.py


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        # get resnet model
        # weights = torchvision.models.ResNet18_Weights.DEFAULT
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.resnet = torchvision.models.resnet50(weights=weights)

        self.fc_in_features = self.resnet.fc.in_features
        # remove fc layer from resnet
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate image features
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        # get resnet model
        # weights = torchvision.models.ResNet18_Weights.DEFAULT
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.resnet = torchvision.models.resnet50(weights=weights)

        self.fc_in_features = self.resnet.fc.in_features

        # remove fc layer from resnet
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )

        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input):
        output = self.forward_once(input)
        output = self.fc(output)
        output = nn.functional.normalize(output, p=2)
        return output
