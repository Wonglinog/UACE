from typing import Callable, Tuple

import torchvision.transforms as transforms


def mnist_transform() -> Callable:
    """PIL Image to Tensor transform for MNIST, with standardization

    Returns:
        transform function
    """

    # Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    mean = (0.1307,)
    std = (0.3081,)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    return transform


def cifar_transform(
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    augment: bool = False,
) -> Callable:
    """PIL Image to Tensor transform for CIFAR,
    with standardization and data augmentation
    Args:
        augment: if True, adds random horizontal flip and random cropping
        mean: RGB channels mean
        std: RGB channels standard deviation


    Returns:
        transform function
    """
    if augment:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    return transform


def cifar10_transform(augment: bool = False) -> Callable:
    """PIL Image to Tensor transform for CIFAR-10,
    with standardization and data augmentation

    Args:
        augment: if True, adds random horizontal flip and random cropping

    Returns:
        transform function
    """
    # Source: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)

    return cifar_transform(mean, std, augment)


def cifar100_transform(augment: bool = False) -> Callable:
    """PIL Image to Tensor transform for CIFAR-100,
    with standardization and data augmentation

    Args:
        augment: if True, adds random horizontal flip and random cropping

    Returns:
        transform function
    """
    # Source: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    return cifar_transform(mean, std, augment)

def clothing1m_transform(augment: bool = False) -> Callable:
    """PIL Image to Tensor transform for CIFAR-100,
    with standardization and data augmentation

    Args:
        augment: if True, adds random horizontal flip and random cropping

    Returns:
        transform function
    """
    # Source: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    if augment:
        transform = transforms.Compose(
            [
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(20),
            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ]
        )
    else:
        transform = transforms.Compose(
            [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ]
        )
    return transform



