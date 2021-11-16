
from collections.abc import Iterable
import numpy as np
import torch

__all__ = [
    "is_list_or_tuple", "assert_numpy", "as_numpy",
    "unpack_tensor_tuple", "pack_tensor_in_tuple",
    "pack_tensor_in_list",
]


def is_list_or_tuple(x):
    return isinstance(x, list) or isinstance(x, tuple)


def assert_numpy(x, arr_type=None):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
    if is_list_or_tuple(x):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    if arr_type is not None:
        x = x.astype(arr_type)
    return x


def as_numpy(tensor):
    """convert tensor to numpy"""
    return torch.as_tensor(tensor).detach().cpu().numpy()


def unpack_tensor_tuple(seq):
    """unpack a tuple containing one tensor to a tensor"""
    if isinstance(seq, torch.Tensor):
        return seq
    else:
        if len(seq) == 1:
            return seq[0]
        else:
            return (*seq, )


def pack_tensor_in_tuple(seq):
    """pack a tensor into a tuple of Tensor of length 1"""
    if isinstance(seq, torch.Tensor):
        return seq,
    elif isinstance(seq, Iterable):
        return (*seq, )
    else:
        return seq


def pack_tensor_in_list(seq):
    """pack a tensor into a list of Tensor of length 1"""
    if isinstance(seq, torch.Tensor):
        return [seq]
    elif isinstance(seq, Iterable):
        return list(seq)
    else:
        return seq
