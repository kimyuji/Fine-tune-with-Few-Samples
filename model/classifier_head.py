from torch import nn
import torch


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, params):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class TwoLayerMLPClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, params):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# class AdaptiveLinearClassifier(nn.Module):
#     def __init__(self, params):
#         super(AdaptiveLinearClassifier).__init__()
#         self.model = LinearClassifier(512, 5, params)
#         self.adaptive_mask = nn.Parameter(torch.tensor(2., requires_grad = True))

#     def forward(self, x):
#         feat_mask = torch.ones_like(x)
#         mins = torch.topk(torch.abs(x), k=int(self.adaptive_mask.data), dim=1, largest=False).indices
#         for i in range(len(mins)):
#             feat_mask[i][mins[i]] = 0
#         x = x * feat_mask
#         x = self.model(x)
#         return x
    



CLASSIFIER_HEAD_CLASS_MAP = {
    # 'adaptivelinear' : AdaptiveLinearClassifier,
    'linear': LinearClassifier,
    'two_layer_mlp': TwoLayerMLPClassifier
}

def get_classifier_head_class(key):
    if key in CLASSIFIER_HEAD_CLASS_MAP:
        return CLASSIFIER_HEAD_CLASS_MAP[key]
    else:
        raise ValueError('Invalid classifier head specifier: {}'.format(key))
