
import numpy as np
from bgflow.factory.tensor_info import ShapeDictionary, TensorInfo, BONDS, ANGLES, TORSIONS, FIXED

def test_shape_info(crd_trafo):
    shape_info = ShapeDictionary.from_coordinate_transform(crd_trafo)

    for key in [BONDS, ANGLES, TORSIONS]:
        assert shape_info[key] == (len(crd_trafo.z_matrix), )
    assert shape_info[FIXED][0] == 3*len(crd_trafo.fixed_atoms)
    assert not (shape_info.is_circular([BONDS, TORSIONS])[: shape_info[BONDS][0]]).any()
    assert (shape_info.is_circular([BONDS, TORSIONS])[shape_info[BONDS][0]:]).all()
    assert shape_info.is_circular().sum() == shape_info[TORSIONS][0]
    assert (
        shape_info.circular_indices([FIXED, TORSIONS])
        == np.arange(shape_info[FIXED][0], shape_info[FIXED][0]+shape_info[TORSIONS][0])
    ).all()
    assert (
        shape_info.circular_indices()
        == np.arange(shape_info[BONDS][0]+shape_info[ANGLES][0],
                     shape_info[BONDS][0]+shape_info[ANGLES][0]+shape_info[TORSIONS][0]
                     )
    ).all()
    assert shape_info.dim_all([BONDS, TORSIONS]) == shape_info[BONDS][0] + shape_info[TORSIONS][0]
    assert shape_info.dim_all() == 66
    assert shape_info.dim_circular([ANGLES, BONDS]) == 0
    assert shape_info.dim_circular() == shape_info[TORSIONS][0]
    assert shape_info.dim_noncircular([ANGLES, BONDS]) == shape_info[ANGLES][0] + shape_info[BONDS][0]
    assert shape_info.dim_noncircular() == 66 - shape_info[TORSIONS][0]

    assert shape_info.dim_cartesian([ANGLES, BONDS]) == 0
    assert shape_info.dim_cartesian([FIXED]) == shape_info[FIXED][0]
    assert shape_info.dim_noncartesian([ANGLES, BONDS]) == shape_info[ANGLES][0] + shape_info[BONDS][0]
    assert shape_info.dim_noncartesian([FIXED]) == 0
    assert not (shape_info.is_cartesian([BONDS, FIXED])[: shape_info[BONDS][0]]).any()
    assert (
        shape_info.cartesian_indices()
        == np.arange(shape_info[BONDS][0]+shape_info[ANGLES][0]+shape_info[TORSIONS][0],
                     shape_info[BONDS][0]+shape_info[ANGLES][0]+shape_info[TORSIONS][0]+shape_info[FIXED][0]
                     )
    ).all()




def test_shape_info_insert():
    shape_info = ShapeDictionary()
    for i in range(5):
        shape_info[i] = (i, )
    shape_info.insert(100, 2, (100, ))
    assert list(shape_info) == [0, 1, 100, 2, 3, 4]
    assert list(shape_info.values()) == [(i, ) for i in [0, 1, 100, 2, 3, 4]]


def test_shape_info_split_merge():
    shape_info = ShapeDictionary()
    for i in range(8):
        shape_info[i] = (i, )
    shape_info.split(4, into=("a", "b"), sizes=(1, 3))
    assert list(shape_info) == [0, 1, 2, 3, "a", "b", 5, 6, 7]
    assert list(shape_info.values()) == [(i, ) for i in [0, 1, 2, 3, 1, 3, 5, 6, 7]]

    shape_info.merge(("a", "b"), to=4)
    assert list(shape_info) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(shape_info.values()) == [(i, ) for i in [0, 1, 2, 3, 4, 5, 6, 7]]
