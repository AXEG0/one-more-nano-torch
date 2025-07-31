import math
from typing import List, Tuple

class Storage:
    def __init__(self, data: List[float], shape: Tuple[int, ...], strides: Tuple[int, ...]):
        self.data = data
        self.shape = shape
        self.strides = strides

    @staticmethod
    def zeros(shape: Tuple[int, ...]) -> 'Storage':
        size = math.prod(shape)
        return Storage([0.0] * size, shape, Storage._get_strides(shape))

    @staticmethod
    def randn(shape: Tuple[int, ...]) -> 'Storage':
        import random
        size = math.prod(shape)
        return Storage([random.random() for _ in range(size)], shape, Storage._get_strides(shape))

    @staticmethod
    def _get_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
