import torchvision

# models provided by torchvision
models = {"resnet18": torchvision.models.resnet18,
          "resnet34": torchvision.models.resnet34,
          "resnet50": torchvision.models.resnet50,
          "resnet101": torchvision.models.resnet101,
          "resnet152": torchvision.models.resnet152,
          "resnext50_32x4d": torchvision.models.resnext50_32x4d,
          "resnext101_32x8d": torchvision.models.resnext101_32x8d,
          "wide_resnet50_2": torchvision.models.wide_resnet50_2,
          "wide_resnet101_2": torchvision.models.wide_resnet101_2}
