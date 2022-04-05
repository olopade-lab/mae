# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# Parts modified from timm
# --------------------------------------------------------

import PIL
from maicara.data.transforms import ToFloat
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
from timm.data.transforms_factory import (
    transforms_imagenet_eval,
    transforms_imagenet_train,
    transforms_noaug_train,
)
from torchvision import transforms


def create_transform(
    img_size,
    is_training=False,
    use_prefetcher=False,
    no_aug=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="bilinear",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    crop_pct=None,
    tf_preprocessing=False,
    separate=False,
):

    if isinstance(img_size, tuple):
        img_size = img_size[-2:]
    else:
        img_size = img_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform

        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=interpolation
        )
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            transform = transforms_noaug_train(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
            )
        elif is_training:
            transform = transforms_imagenet_train(
                img_size,
                scale=scale,
                ratio=ratio,
                hflip=hflip,
                vflip=vflip,
                color_jitter=color_jitter,
                auto_augment=auto_augment,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits,
                separate=separate,
            )
        else:
            assert (
                not separate
            ), "Separate transforms not supported for validation preprocessing"
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                crop_pct=crop_pct,
            )

    return transform


def build_transform(
    is_train, args, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):
    if is_train:
        transform = create_transform(
            img_size=args.img_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.img_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.img_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.img_size))

    t.append(transforms.ToTensor())
    # t.append(ToFloat())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
