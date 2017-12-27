from deeppavlov.core.registrable import Registrable
import random


class DatasetReader:
    """
    A ``DatasetReader`` reads data from some location and constructs a dataset.
    """
    @staticmethod
    def read(data_path: str, *args, **kwargs):
        """
        Read a file from a path and returns data as list with training instances.
        """
        raise NotImplementedError


class DatasetProvider(Registrable):
    def split(self, *args, **kwargs):
        pass

    def __init__(self, data, seed, *args, **kwargs):
        r""" Dataset takes a dict with fields 'train', 'test', 'valid'. A list of samples (pairs x, y) is stored
        in each field.
        Args:
            data: list of (x, y) pairs. Each pair is a sample from the dataset. x as well as y can be a tuple
                of different input features.
            seed (int): random seed for data shuffling. Defaults to None
        """

        rs = random.getstate()
        random.seed(seed)
        self.random_state = random.getstate()
        random.setstate(rs)

        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

    def batch_generator(self, batch_size, data_type = 'train'):
        r"""This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
         batches
        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'
        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
        """
        if batch_size == -1:
            yield from self.iter_all(data_type)
        else:
            data = self.data[data_type]
            data_len = len(data)
            order = list(range(data_len))

            rs = random.getstate()
            random.setstate(self.random_state)
            random.shuffle(order)
            self.random_state = random.getstate()
            random.setstate(rs)

            for i in range((data_len - 1) // batch_size + 1):
                yield list(zip(*[data[o] for o in order[i*batch_size:(i+1)*batch_size]]))

    def iter_all(self, data_type='train'):
        r"""Iterate through all data. It can be used for building dictionary or
        Args:
            data_type (str): can be either 'train', 'test', or 'valid'
        Returns:
            samples_gen: a generator, that iterates through the all samples in the selected data type of the dataset
        """
        data = self.data[data_type]
        for sample in data:
            yield (sample,)