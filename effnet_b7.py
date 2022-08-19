
from torch import nn
from efficientnet_pytorch import EfficientNet


class modified_EfficientNet(nn.Module):
    def __init__(self):
        super(modified_EfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.model._fc = nn.Linear(self.model._fc.in_features, 1)
        self.model._fc_2 = nn.Sequential(nn.Linear(512, 256),
                                         nn.BatchNorm1d(256),
                                         nn.Linear(256, 128),
                                         nn.BatchNorm1d(128),
                                         nn.Linear(128, 1),
                                         )
        self.model._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.model._dropout = nn.Dropout(p=0.2)
        self.model._relu = nn.ReLU()

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.model._fc(x)
        x = self.model._dropout(x)
        x = self.model._fc_2(x)
        #to neglect the negative outputs
        x = self.model._relu(x)
        return x.squeeze(1)
