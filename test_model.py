from vit_model import VisionTransformer
import torchvision
import torch

if __name__ == '__main__':
    model = VisionTransformer()
    X = torch.rand((5, 3, 224, 224))
    out = model(X)
    pass