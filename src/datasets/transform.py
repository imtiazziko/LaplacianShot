import torchvision.transforms as transforms
from .imagejitter import ImageJitter

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)

def without_augment(size=84, enlarge=False):
    if enlarge:
        resize = int(size*256./224.)
    else:
        resize = size
    return transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

def with_augment(size=84, disable_random_resize=False, jitter=False):
    # Added Jitter.
    if disable_random_resize:
        transform_funcs = [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        transform_funcs = [
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    if jitter:
        jitter_func = ImageJitter(jitter_param)
        transform_funcs.insert(1, jitter_func)

    transform = transforms.Compose(transform_funcs)

    return transform