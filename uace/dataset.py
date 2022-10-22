import collections
from typing import Callable, Optional, Tuple
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST,FashionMNIST
from numpy.testing import assert_array_almost_equal
from PIL import Image
import os


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    
    other_class = np.random.choice(other_class_list)
    return other_class


class NoisyMNIST(MNIST):
    """Extends `torchvision.datasets.MNIST
    <https://pytorch.org/docs/stable/torchvision/datasets.html#mnist>`_
    class by corrupting the labels with a fixed probability
    """

    num_classes = 10

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        corrupt_prob: float = 0.0,
        asym: int= 0,
        noise_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.corrupt_prob = corrupt_prob
        self.noise_seed = noise_seed
        self.asym = asym
        self.true_target = torch.zeros_like(self.targets)
        self._add_label_noise()

        #noise_label = torch.load('./data/CIFAR-10_human.pt')
        #clean_label = noise_label['clean_label']
        #worst_label = noise_label['worse_label']

    def _add_label_noise(self) -> None:
        self.true_target = copy.deepcopy(self.targets)
        if self.corrupt_prob < 0 or self.corrupt_prob > 1:
            raise ValueError(f"Invalid noise probability: {self.corrupt_prob}")

        if self.corrupt_prob == 0:
            return

        if self.noise_seed is not None:
            np.random.seed(self.noise_seed)

        if self.asym == 0:
            p = np.ones((len(self.targets), self.num_classes))
            p = p * (self.corrupt_prob / (self.num_classes - 1))
            p[np.arange(len(self.targets)), self.targets] = 1 - self.corrupt_prob


            print("symmetric",p)
            for i in range(len(self.targets)):
                self.targets[i] = np.random.choice(self.num_classes, p=p[i])
        elif self.asym == 1:

            nb_classes = self.num_classes
            noise_matrix = np.eye(nb_classes)
            n = self.corrupt_prob

            noise_matrix[7,7],noise_matrix[7,1] = 1-n, n
            noise_matrix[2,2],noise_matrix[2,7] = 1-n, n
            noise_matrix[5,5],noise_matrix[5,6] = 1-n, n
            noise_matrix[6,6],noise_matrix[6,5] = 1-n, n
            noise_matrix[3,3],noise_matrix[3,8] = 1-n, n

            print("asymmetric",noise_matrix)
            p = np.ones((len(self.targets), self.num_classes)) 
            p[np.arange(len(self.targets)), :]=noise_matrix[self.targets]

            #print("p",p[0,19],p[0,15])

            for i in range(len(self.targets)):
                self.targets[i] = np.random.choice(self.num_classes, p=p[i])

        else:

            assert 1 == 0 

        print("noise_rate",np.sum(np.array(self.true_target)!=np.array(self.targets))/len(self.true_target))


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target,true_target = self.data[index], int(self.targets[index]),int(self.true_target[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            true_target = self.target_transform(true_target)

        return img, target,true_target,index


class NoisyFASHIONMNIST(FashionMNIST):
    """Extends `torchvision.datasets.MNIST
    <https://pytorch.org/docs/stable/torchvision/datasets.html#mnist>`_
    class by corrupting the labels with a fixed probability
    """

    num_classes = 10

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        corrupt_prob: float = 0.0,
        asym: int= 0,
        noise_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.corrupt_prob = corrupt_prob
        self.noise_seed = noise_seed
        self.asym = asym
        self.true_target = torch.zeros_like(self.targets)
        self._add_label_noise()

        #noise_label = torch.load('./data/CIFAR-10_human.pt')
        #clean_label = noise_label['clean_label']
        #worst_label = noise_label['worse_label']

    def _add_label_noise(self) -> None:
        self.true_target = copy.deepcopy(self.targets)
        if self.corrupt_prob < 0 or self.corrupt_prob > 1:
            raise ValueError(f"Invalid noise probability: {self.corrupt_prob}")

        if self.corrupt_prob == 0:
            return

        if self.noise_seed is not None:
            np.random.seed(self.noise_seed)

        if self.asym == 0:
            p = np.ones((len(self.targets), self.num_classes))
            p = p * (self.corrupt_prob / (self.num_classes - 1))
            p[np.arange(len(self.targets)), self.targets] = 1 - self.corrupt_prob


            print("symmetric",p)
            for i in range(len(self.targets)):
                self.targets[i] = np.random.choice(self.num_classes, p=p[i])
        elif self.asym == 1:

            nb_classes = self.num_classes
            noise_matrix = np.eye(nb_classes)
            n = self.corrupt_prob

            noise_matrix[7,7],noise_matrix[7,1] = 1-n, n
            noise_matrix[2,2],noise_matrix[2,7] = 1-n, n
            noise_matrix[5,5],noise_matrix[5,6] = 1-n, n
            noise_matrix[6,6],noise_matrix[6,5] = 1-n, n
            noise_matrix[3,3],noise_matrix[3,8] = 1-n, n

            print("asymmetric",noise_matrix)
            p = np.ones((len(self.targets), self.num_classes)) 
            p[np.arange(len(self.targets)), :]=noise_matrix[self.targets]

            #print("p",p[0,19],p[0,15])

            for i in range(len(self.targets)):
                self.targets[i] = np.random.choice(self.num_classes, p=p[i])

        else:

            assert 1 == 0 

        print("noise_rate",np.sum(np.array(self.true_target)!=np.array(self.targets))/len(self.true_target))


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target,true_target = self.data[index], int(self.targets[index]),int(self.true_target[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            true_target = self.target_transform(true_target)

        return img, target,true_target,index


class NoisyCIFAR10(CIFAR10):
    """Extends `torchvision.datasets.CIFAR10
    <https://pytorch.org/docs/stable/torchvision/datasets.html#cifar>`_
    class by corrupting the labels with a fixed probability
    """

    num_classes = 10

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        corrupt_prob: float = 0.0,
        asym: int= 0,
        noise_seed: Optional[int] = None,
    ) -> None:

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.corrupt_prob = corrupt_prob
        self.noise_seed = noise_seed
        self.asym = asym
        self.true_target = copy.deepcopy(self.targets)

        print("--------self_asym--------",self.asym)
        self._add_label_noise()

    def _add_label_noise(self) -> None:
        
        print("+++++++++++++++++++")

        if self.corrupt_prob < 0 or self.corrupt_prob > 1:
            raise ValueError(f"Invalid noise probability: {self.corrupt_prob}")

        if self.corrupt_prob == 0:
            return

        if self.noise_seed is not None:
            np.random.seed(self.noise_seed)

        # if self.asym == 0:
        #     p = np.ones((len(self.targets), self.num_classes))
        #     p = p * (self.corrupt_prob / (self.num_classes - 1))
        #     p[np.arange(len(self.targets)), self.targets] = 1 - self.corrupt_prob

        #     print("symmetric",p)

        #     for i in range(len(self.targets)):
        #         self.targets[i] = np.random.choice(self.num_classes, p=p[i])

        # elif self.asym == 1:
        #     #source_class = [9, 2, 3, 5, 4]
        #     #target_class = [1, 0, 5, 3, 7]

        #     nb_classes = self.num_classes
        #     noise_matrix = np.eye(nb_classes)
        #     n = self.corrupt_prob

        #     noise_matrix[9,9],noise_matrix[9,1] = 1-n, n
        #     noise_matrix[2,2],noise_matrix[2,0] = 1-n, n
        #     noise_matrix[3,3],noise_matrix[3,5] = 1-n, n
        #     noise_matrix[5,5],noise_matrix[5,3] = 1-n, n
        #     noise_matrix[4,4],noise_matrix[4,7] = 1-n, n

        #     print("asymmetric",noise_matrix)
        #     p = np.ones((len(self.targets), self.num_classes)) 
        #     p[np.arange(len(self.targets)), :]=noise_matrix[self.targets]

        #     #print("p",p[0,19],p[0,15])

        #     for i in range(len(self.targets)):
        #         self.targets[i] = np.random.choice(self.num_classes, p=p[i])


        # if self.asym == 0:
        #     p = np.ones((len(self.targets), self.num_classes))
        #     p = p * (self.corrupt_prob / (self.num_classes - 1))
        #     p[np.arange(len(self.targets)), self.targets] = 1 - self.corrupt_prob

        #     print("symmetric",p)

        #     for i in range(len(self.targets)):
        #         self.targets[i] = np.random.choice(self.num_classes, p=p[i])

        if self.asym == 0:
            assert self.corrupt_prob > 0
            n_samples = len(self.targets)
            n_noisy = int(self.corrupt_prob * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

        elif self.asym == 1:
            # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(self.corrupt_prob * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

        else:

            print("------------*******************---------------")
            #/home/phd-wang.lin/workspace/pytorch/phuber1/data/
            noise_label = torch.load('/home/phd-wang.lin/workspace/pytorch/UACE/data/CIFAR-10_human.pt')

            if self.asym == 2:
                self.targets = noise_label['worse_label']
                print("worse_label",self.asym)
            elif self.asym == 3:
                self.targets = noise_label['aggre_label']
                print("aggre_label",self.asym)
            elif self.asym == 4:
                self.targets = noise_label['random_label1']
                print("random_label1",self.asym)
            elif self.asym == 5:
                self.targets = noise_label['random_label2']
                print("random_label2",self.asym)
            elif self.asym == 6:
                self.targets = noise_label['random_label3']
                print("random_label3",self.asym)
            else:
                assert 1 == 0

            #worst_label = noise_label['worse_label']
            #
            print("real-world-noise")

            print("---",self.targets)
            return 


        print("noise_rate",np.sum(np.array(self.true_target)!=np.array(self.targets))/len(self.true_target))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target,true_target = self.data[index], self.targets[index], self.true_target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            true_target = self.target_transform(true_target)


        return img, target,true_target,index


class NoisyCIFAR100(CIFAR100):
    """Extends `torchvision.datasets.CIFAR100
    <https://pytorch.org/docs/stable/torchvision/datasets.html#cifar>`_
    class by corrupting the labels with a fixed probability
    """

    num_classes = 100

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        corrupt_prob: float = 0.0,
        asym: int= 0,
        noise_seed: Optional[int] = None,
    ) -> None:

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.corrupt_prob = corrupt_prob
        self.noise_seed = noise_seed
        self.asym = asym
        self.true_target = copy.deepcopy(self.targets)
        self._add_label_noise()
        

    def _add_label_noise(self) -> None:
        

        if self.corrupt_prob < 0 or self.corrupt_prob > 1:
            raise ValueError(f"Invalid noise probability: {self.corrupt_prob}")

        if self.corrupt_prob == 0:
            return

        if self.noise_seed is not None:
            np.random.seed(self.noise_seed)

        if self.asym == 0:
            p = np.ones((len(self.targets), self.num_classes))
            p = p * (self.corrupt_prob / (self.num_classes - 1))
            p[np.arange(len(self.targets)), self.targets] = 1 - self.corrupt_prob
            print("---------------------",self.asym,p)
            for i in range(len(self.targets)):
                self.targets[i] = np.random.choice(self.num_classes, p=p[i])
        elif self.asym == 1:
            nb_classes = self.num_classes
            noise_matrix = np.eye(nb_classes)
            n = self.corrupt_prob
            nb_superclasses = 20
            nb_subclasses = 5

            if n > 0.0:
                for i in np.arange(nb_superclasses):
                    init, end = i * nb_subclasses, (i+1) * nb_subclasses
                    noise_matrix[init:end, init:end] = build_for_cifar100(nb_subclasses, n)
                #print("-------",noise_matrix,self.targets[0],self.targets[1],self.targets[2])

            p = np.ones((len(self.targets), self.num_classes)) 
            p[np.arange(len(self.targets)), :]=noise_matrix[self.targets]

            #print("p",p[0,19],p[0,15])

            for i in range(len(self.targets)):
                self.targets[i] = np.random.choice(self.num_classes, p=p[i])
        elif self.asym == 2:
            noise_label = torch.load('/home/phd-wang.lin/workspace/pytorch/UACE/data/CIFAR-100_human.pt')
            self.targets = noise_label['noisy_label']

            print("real-world-noise")

        else:
            assert 1 == 0


        print("noise_rate",np.sum(np.array(self.true_target)!=np.array(self.targets))/len(self.true_target))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target,true_target = self.data[index], self.targets[index], self.true_target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            true_target = self.target_transform(true_target)


        return img, target,true_target,index





def split_dataset(dataset: Dataset, split: float, seed: int) -> Tuple[Subset, Subset]:
    """Splits dataset into a train / val set based on a split value and seed

    Args:
        dataset: dataset to split
        split: The proportion of the dataset to include in the validation split,
            must be between 0 and 1.
        seed: Seed used to generate the split

    Returns:
        Subsets of the input dataset

    """
    # Verify that the dataset is Sized
    if not isinstance(dataset, collections.abc.Sized):
        raise ValueError("Dataset is not Sized!")

    if not (0 <= split <= 1):
        raise ValueError(f"Split value must be between 0 and 1. Value: {split}")

    val_length = int(len(dataset) * split)
    train_length = len(dataset) - val_length
    splits = random_split(
        dataset,
        [train_length, val_length],
        generator=torch.Generator().manual_seed(seed),
    )
    return splits


def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class




class Clothing1MDataset:
    def __init__(self, 
                 root: str,
                 train: bool = True, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 corrupt_prob: float = 0.0,
                 asym: int= 0,
                 noise_seed: Optional[int] = None,
                 ) -> None:
        self.data_root = root
        if train:
            file_path = os.path.join(self.data_root, 'annotations/noisy_train_key_list.txt')
            label_path = os.path.join(self.data_root, 'annotations/my_train_label.txt')
        else :
            file_path = os.path.join(self.data_root, 'annotations/clean_test_key_list.txt')
            label_path = os.path.join(self.data_root, 'annotations/my_test_label.txt')

        with open(file_path) as fid:
            image_list = [os.path.join(self.data_root,line.strip()) for line in fid.readlines()]

        with open(label_path) as fid:
            label_list = [int(line.strip()) for line in fid.readlines()]

        self.imlist = [(image_list[i],label_list[i]) for i in range(len(image_list))]
        print(self.imlist[1])
        self.transform = transform

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = Image.open(impath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target, target, index




