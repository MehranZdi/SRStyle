import torch.nn as nn

class AdaIN(nn.Module):
    '''This class implements the AdaIN layer(Adaptive Invariant Neural Network)'''
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, content_features, style_features):
        # Calculate mean and std for content and style features
        content_mean = content_features.mean(dim=(2, 3), keepdim=True)              # dim=(2, 3) refers to the height and the width of the feature map in Pytorch
        content_std = content_features.std(dim=(2, 3), keepdim=True) + self.eps
        style_mean = style_features.mean(dim=(2, 3), keepdim=True)
        style_std = style_features.std(dim=(2, 3), keepdim=True) + self.eps

        # Normalize content features
        normalized_content_features = (content_features - content_mean) / content_std

        # Scale and shift with style features
        stylized_features = style_std * normalized_content_features + style_mean

        return stylized_features
