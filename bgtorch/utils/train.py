import numpy as np


class IndexBatchIterator(object):
    def __init__(self, n_elems, n_batch):
        """
            Produces batches of length `n_batch` of an index set
            `[1, ..., n_elems]` which are sampled randomly without
            replacement.

            If `n_elems` is not a multiple of `n_batch` the last sampled
            batch will be truncated.

            After the iteration throw `StopIteration` its random seed
            will be reset.

            Parameters:
            -----------
            n_elems : Integer
                Number of elements in the index set.
            n_batch : Integer
                Number of batch elements sampled.

        """
        self._indices = np.arange(n_elems)
        self._n_elems = n_elems
        self._n_batch = n_batch
        self._pos = 0
        self._reset()

    def _reset(self):
        self._pos = 0
        np.random.shuffle(self._indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos >= self._n_elems:
            self._reset()
            raise StopIteration
        n_collected = min(self._n_batch, self._n_elems - self._pos)
        batch = self._indices[self._pos : self._pos + n_collected]
        self._pos = self._pos + n_collected
        return batch

    def __len__(self):
        return self._n_elems // self._n_batch

    def next(self):
        return self.__next__()
