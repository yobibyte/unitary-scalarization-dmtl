# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/

import torch
from torchvision import transforms
from supervised_experiments.loaders.multi_mnist_loader import MNIST
from supervised_experiments.loaders.cityscapes_loader import CityScape
from supervised_experiments.loaders.celeba_loader import CELEBA


def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(dataset, batch_size, configs, generator=None, worker_init_fn=None, train=True):

    if 'mnist' in dataset:
        if train:
            # Return training + validation split for training loop.
            train_dst = MNIST(root=configs['mnist']['path'], split="train", download=True, transform=global_transformer())
            train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=4,
                                                       generator=generator, worker_init_fn=worker_init_fn)

            val_dst = MNIST(root=configs['mnist']['path'], split="val", download=True, transform=global_transformer())
            val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=4,
                                                     generator=generator, worker_init_fn=worker_init_fn)
            return train_loader, val_loader
        else:
            # Return test split only for evaluation of a stored model.
            test_dst = MNIST(root=configs['mnist']['path'], split="test", download=True, transform=global_transformer())
            test_loader = torch.utils.data.DataLoader(test_dst, batch_size=100, shuffle=True, num_workers=4,
                                                      generator=generator, worker_init_fn=worker_init_fn)
            return test_loader


    if 'cityscapes' in dataset:

        if train:
            train_loader = torch.utils.data.DataLoader(
                dataset=CityScape(root=configs['cityscapes']['path'], mode='train', augmentation=False),
                batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                generator=generator, worker_init_fn=worker_init_fn)

            val_loader = torch.utils.data.DataLoader(
                dataset=CityScape(root=configs['cityscapes']['path'], mode='val', augmentation=False),
                batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                generator=generator, worker_init_fn=worker_init_fn)

            return train_loader, val_loader

        else:
            return torch.utils.data.DataLoader(
                dataset=CityScape(root=configs['cityscapes']['path'], mode='test', augmentation=False),
                batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                generator=generator, worker_init_fn=worker_init_fn)

    if 'celeba' in dataset:
        if train:
            # Return training + validation split for training loop.
            train_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='train',
                               img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
                               augmentations=None)
            val_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val',
                             img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
                             augmentations=None)

            train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=4,
                                                       generator=generator, worker_init_fn=worker_init_fn)
            val_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size, num_workers=4,
                                                     generator=generator, worker_init_fn=worker_init_fn)
            return train_loader, val_loader
        else:
            # Return test split only for evaluation of a stored model.
            test_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='test',
                             img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
                             augmentations=None)
            test_loader = torch.utils.data.DataLoader(test_dst, batch_size=batch_size, num_workers=4,
                                                      generator=generator, worker_init_fn=worker_init_fn)
            return test_loader
