import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, output, labels):
        return self.loss(output, labels)
