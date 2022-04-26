import torchvision
from models.e2_wide_resnet import e2wrn28_10R, e2wrn28_7R

models = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
    "resnext50_32x4d": torchvision.models.resnext50_32x4d,
    "resnext101_32x8d": torchvision.models.resnext101_32x8d,
    "wide_resnet50_2": torchvision.models.wide_resnet50_2,
    "wide_resnet101_2": torchvision.models.wide_resnet101_2,
    "densenet": torchvision.models.DenseNet,
    "e2_wide_resnet28_10R": e2wrn28_10R,
    "e2_wide_resnet28_7R": e2wrn28_7R
}

def get_model(model_type, num_classes, args):
    model_args = {"num_classes": num_classes}
    densenet_args = {"growth_rate": args.growth_rate,
                     "block_config": (3, 3, 3),
                     "num_init_features": args.num_init_features}

    if model_type == "densenet":
        model_args = {**model_args, **densenet_args}

    return models[model_type](**model_args)

