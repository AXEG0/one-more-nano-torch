from src.core.storage import Storage

def matmul(a: Storage, b: Storage) -> Storage:
    if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Invalid shapes for matmul")

    c_shape = (a.shape[0], b.shape[1])
    c = Storage.zeros(c_shape)

    for i in range(c_shape[0]):
        for j in range(c_shape[1]):
            s = 0
            for k in range(a.shape[1]):
                s += a.data[i * a.strides[0] + k * a.strides[1]] * b.data[k * b.strides[0] + j * b.strides[1]]
            c.data[i * c.strides[0] + j * c.strides[1]] = s

    return c
