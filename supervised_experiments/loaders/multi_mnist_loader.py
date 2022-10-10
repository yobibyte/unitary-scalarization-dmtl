#Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import errno
import numpy as np
import torch
import codecs


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    multi_training_file = 'multi_training.pt'
    multi_validation_file = 'multi_validation.pt'
    multi_test_file = 'multi_test.pt'

    def __init__(self, root, split="train", transform=None, download=False):
        assert split in ["train", "val", "test"]

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        if download:
            self.download()

        if not self._check_multi_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download MNIST and generate a random MultiMNIST')

        if self.split == "train":
            self.train_data, self.train_labels_l, self.train_labels_r = torch.load(
                os.path.join(self.root, self.processed_folder, self.multi_training_file))
        elif self.split == "val":
            self.validation_data, self.validation_labels_l, self.validation_labels_r = torch.load(
                os.path.join(self.root, self.processed_folder, self.multi_validation_file))
        else:
            self.test_data, self.test_labels_l, self.test_labels_r = torch.load(
                os.path.join(self.root, self.processed_folder, self.multi_test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == "train":
            img, target_l, target_r = self.train_data[index], self.train_labels_l[index], self.train_labels_r[index]
        elif self.split == "val":
            img, target_l, target_r = self.validation_data[index], self.validation_labels_l[index], \
                                      self.validation_labels_r[index]
        else:
            img, target_l, target_r = self.test_data[index], self.test_labels_l[index], self.test_labels_r[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        return img, target_l, target_r

    def __len__(self):
        if self.split == "train":
            return len(self.train_data)
        elif self.split == "val":
            return len(self.validation_data)
        else:
            return len(self.test_data)

    def _check_multi_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_test_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_validation_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_multi_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        # Create train-set images from MNIST's original training set.
        mnist_ims, multi_mnist_ims, left_idx, right_idx = create_multimnist_images(
            os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'), split="train")
        mnist_labels, multi_mnist_labels_l, multi_mnist_labels_r = create_multimnist_labels(
            os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'), left_idx, right_idx)

        # Create validation set images from MNIST's original training set.
        vmnist_ims, vmulti_mnist_ims, left_idx, right_idx = create_multimnist_images(
            os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'), split="val")
        vmnist_labels, vmulti_mnist_labels_l, vmulti_mnist_labels_r = create_multimnist_labels(
            os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'), left_idx, right_idx)

        # Create test set images from MNIST's original test set (the second image to be overlapped is randomly chosen).
        tmnist_ims, tmulti_mnist_ims, left_idx, right_idx = create_multimnist_images(
            os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'), split="test")
        tmnist_labels, tmulti_mnist_labels_l, tmulti_mnist_labels_r = create_multimnist_labels(
            os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'), left_idx, right_idx)

        multi_mnist_training_set = (multi_mnist_ims, multi_mnist_labels_l, multi_mnist_labels_r)
        multi_mnist_validation_set = (vmulti_mnist_ims, vmulti_mnist_labels_l, vmulti_mnist_labels_r)
        multi_mnist_test_set = (tmulti_mnist_ims, tmulti_mnist_labels_l, tmulti_mnist_labels_r)

        with open(os.path.join(self.root, self.processed_folder, self.multi_training_file), 'wb') as f:
            torch.save(multi_mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_validation_file), 'wb') as f:
            torch.save(multi_mnist_validation_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_test_file), 'wb') as f:
            torch.save(multi_mnist_test_set, f)
        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def create_multimnist_labels(path, left_indices, right_indices):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        nom_length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        length = len(left_indices)
        multi_labels_l = np.zeros(length, dtype=np.long)
        multi_labels_r = np.zeros(length, dtype=np.long)
        for im_id in range(len(left_indices)):
            multi_labels_l[im_id] = parsed[left_indices[im_id]]
            multi_labels_r[im_id] = parsed[right_indices[im_id]]
        return torch.from_numpy(parsed).view(nom_length).long(), \
               torch.from_numpy(multi_labels_l).view(length).long(), \
               torch.from_numpy(multi_labels_r).view(length).long()


def create_multimnist_images(path, split="train"):
    assert split in ["train", "val", "test"]

    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        nom_length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        pv = parsed.reshape(nom_length, num_rows, num_cols)

        if split == "train":
            assert nom_length == 60000, "Need to pass the original MNIST training set to create a training split"
            val_size = 10000  # same size as original test split
            start = 0
            end = nom_length - val_size
        elif split == "val":
            assert nom_length == 60000, "Need to pass the original MNIST training set to create a validation split"
            val_size = 10000  # same size as original test split
            start = nom_length - val_size
            end = nom_length
        else:  # split == "test"
            assert nom_length == 10000, "Need to pass the original MNIST test set to create a test split"
            start = 0
            end = nom_length
        length = end - start
        multi_data = np.zeros((1*length, num_rows, num_cols))

        left_indices = list(range(start, end))
        right_indices = np.random.randint(low=start, high=end, size=end).tolist()

        for im_id in range(len(left_indices)):
            lim = pv[left_indices[im_id], :, :]
            rim = pv[right_indices[im_id], :, :]
            new_im = np.zeros((36, 36))
            new_im[0:28, 0:28] = lim
            new_im[6:34, 6:34] = rim
            new_im[6:28, 6:28] = np.maximum(lim[6:28, 6:28], rim[0:22, 0:22])
            multi_data_im = np.array(PIL.Image.fromarray(new_im).resize((28, 28), resample=PIL.Image.NEAREST))
            multi_data[im_id, :, :] = multi_data_im

        return torch.from_numpy(parsed).view(nom_length, num_rows, num_cols), \
               torch.from_numpy(multi_data).view(length, num_rows, num_cols), left_indices, right_indices


