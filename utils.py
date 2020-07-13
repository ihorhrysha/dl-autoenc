import numpy as np
import matplotlib.pyplot as plt
import math


def diff_numpy(a, b, msg=None):
    """Shows differences between two tensors"""
    if a.shape != b.shape:
        print('Wrong shape!')
        print(a.shape)
        print(b.shape)
    else:
        diff = (np.sum(a - b))**2
        if msg:
            print('%s diff = %1.6f' % (msg, diff.item()))
        else:
            print('diff = %1.6f' % diff.item())


def images2batches(images):
    """Converts images to convenient for batching form"""
    ndata, img_size, _ = images.shape
    return np.reshape(images, (ndata, img_size*img_size))


def batches2images(batches):
    """Converts images to convenient for batching form"""
    ndata, img_size_2 = batches.shape
    img_size = int(math.sqrt(img_size_2))
    return np.reshape(batches, (ndata, img_size, img_size))


def imshow(img):
    """Show image using matplotlib"""
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def init_uniform(a, init_type='xavier'):
    """Makes iniform initialization of weight matrix (please, use
    numpy.random.uniform function or similar"""

    # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    nin, nout = a.shape

    if init_type == 'xavier':
        return np.random.randn(nin, nout) * np.sqrt(1/nin)
    elif init_type == 'he':
        return np.random.randn(nin, nout) * np.sqrt(2/nin)
    else:
        # uniform
        sd = np.sqrt(6.0 / (nin + nout))
        return np.random.uniform(-sd, sd, a.shape)


def relu(m):
    """Implements ReLU activation function"""
    return np.maximum(0, m)


def reduce_images(x, nth):

    x_reduced = x[:, ::nth]

    assert(x_reduced[0, 0] == x[0, 0])

    return x_reduced


def get_random_batch(batches_train, batch_size):
    """Outputs random batch of batch_size"""
    idxs = np.random.choice(batches_train.shape[0], batch_size, replace=False)
    return batches_train[idxs, :]


def scalar_mult(A1, A2):

    m, n = A1.shape
    n1, k = A2.shape

    res = np.zeros((m, k))

    for i_m in range(m):
        for i_n in range(n):
            for i_k in range(k):
                res[i_m, i_k] += A1[i_m, i_n]*A2[i_n, i_k]

    return res


def get_loss(Y_batch, X_batch_train):
    """Claculates sum squared loss"""
    loss = np.sum(np.power(Y_batch-X_batch_train, 2))
    m = Y_batch.shape[0]
    return np.squeeze(loss)/m
