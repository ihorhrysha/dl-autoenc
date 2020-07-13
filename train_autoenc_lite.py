from __future__ import print_function
import numpy as np
import pickle
from utils import batches2images, get_loss, get_random_batch, images2batches, init_uniform, relu, scalar_mult, reduce_images
import time
import matplotlib.pyplot as plt


BATCH_SIZE = 50
UPDATES_NUM = 3000
IMG_SIZE = 15
D = 225  # IMG_SIZE*IMG_SIZE
P = 75  # D /// 3
LEARNING_RATE = 0.001


class EncDecNetLite():
    def __init__(self, use_scalar_mult=False):
        super(EncDecNetLite, self).__init__()
        self.w_in = np.zeros((P, D))
        self.b_in = np.zeros((1, P))

        self.w_rec = np.eye(P)
        self.b_rec = np.zeros((1, P))

        self.w_link = np.zeros((P, P))
        self.b_link = np.zeros((1, P))  # not trainable

        self.w_out = np.zeros((D, P))
        self.b_out = np.zeros((1, D))

    def init(self):
        self.w_in = init_uniform(self.w_in)
        self.w_link = init_uniform(self.w_link)
        self.w_out = init_uniform(self.w_out)

    def _layer_forward(self, z_prev, layer, use_relu=True):
        """One layer's forward pass"""

        self.__dict__['z_prev_'+layer] = z_prev
        b = self.__getattribute__('b_'+layer)
        w = self.__getattribute__('w_'+layer)

        dim_out = w.shape[0]

        # simplification due to np broadcasting
        a = z_prev@w.T + b

        z = relu(a) if use_relu else a

        return (a, z)

    def forward(self, x):

        # B_in = np.matmul(np.ones((BATCH_SIZE, 1)),
        #                  self.b_in.reshape(1, P)) # [20, 75]
        # a_in = np.matmul(x, self.w_in.transpose()) + B_in # [20, 75]
        # z_in_numpy = relu(a_in)

        self.a_in, z_in = self._layer_forward(x, 'in')

        self.a_rec, z_rec = self._layer_forward(z_in, 'rec')

        x_reduce = reduce_images(x, 3)

        self.a_link, z_link = self._layer_forward(
            x_reduce, 'link', use_relu=False)

        z_link_rec = z_link+z_rec

        self.a_out, y = self._layer_forward(z_link_rec, 'out')

        return y  # y

    def _layer_backprop(self, dZ, layer, use_relu=True):
        """One layer's backprop pass"""

        b = self.__getattribute__('b_'+layer)
        w = self.__getattribute__('w_' + layer)
        a = self.__getattribute__('a_' + layer)
        z_prev = self.__getattribute__('z_prev_' + layer)

        dA = np.array(dZ, copy=True)
        if use_relu:
            dA[a <= 0] = 0

        m = z_prev.shape[0]

        # correct mult
        dW = np.dot(dA.T, z_prev)/m
        db = np.sum(dA, axis=0, keepdims=True)/m
        dZ_prev = np.dot(dA, w)

        return dZ_prev, dW, db

    def backprop(self, y, x):

        dw = {}

        dy = 2 * (y - x)

        layer = "out"
        dZ_prev, dw["dw_" + layer], dw["db_" +
                                       layer] = self._layer_backprop(dy, layer)

        layer = "link"
        _, dw["dw_" + layer], dw["db_" +
                                 layer] = self._layer_backprop(dZ_prev, layer, False)

        layer = "rec"
        dZ_prev, _, _ = self._layer_backprop(dZ_prev, layer)

        layer = "in"
        _, dw["dw_" + layer], dw["db_" +
                                 layer] = self._layer_backprop(dZ_prev, layer, False)
        return dw  # dw

    def apply_dw(self, dw):
        """Correct neural network''s weights"""

        # list of trainable params
        param_names = ["w_out", "b_out", "w_link", "w_in", "b_in"]

        for param_name in param_names:
            self.__dict__[param_name] = self.__getattribute__(
                param_name) - LEARNING_RATE*dw["d" + param_name]


# Load train data
images_train = pickle.load(open('images_train.pickle', 'rb'))

# Convert images to batching-friendly format
batches_train = images2batches(images_train)

# Scailing
batches_train = batches_train/255

# Create neural network
neural_network = EncDecNetLite()
# Initialize weights
neural_network.init()

losses = []

# Main cycle
for i in range(UPDATES_NUM):
    # Get random batch for Stochastic Gradient Descent
    X_batch_train = get_random_batch(batches_train, BATCH_SIZE)

    # Forward pass, calculate network''s outputs
    Y_batch = neural_network.forward(X_batch_train)

    # Calculate sum squared loss
    loss = get_loss(Y_batch, X_batch_train)

    # Backward pass, calculate derivatives of loss w.r.t. weights
    dw = neural_network.backprop(Y_batch, X_batch_train)

    # Correct neural network''s weights
    neural_network.apply_dw(dw)

    # Print the loss every 1000 iterations
    if i % 10 == 0:
        print("Cost after iteration {}: {}".format(i, loss))
        losses.append(loss)


# --------------------------------------------------------------------------------------
# plot the loss
plt.plot(losses)
plt.ylabel('cost')
plt.xlabel('epochs')
plt.title("Learning rate =" + str(LEARNING_RATE))
plt.savefig('loss_curve.png')

# --------------------------------------------------------------------------------------
# test images
images_test = pickle.load(open('images_train.pickle', 'rb'))

# unpickle
X_test = images2batches(images_test)

# scale
X_test = X_test / 255

# result images
Y_test = neural_network.forward(X_test)

# scale back
Y_test = (Y_test * 255).round(decimals=0)

result_images_test = batches2images(Y_test)

print_amount = 10

plt.figure(figsize=(10, 3))
for i in range(print_amount):
    plt.subplot(2, print_amount, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_test[i], cmap=plt.cm.gray, vmin=0, vmax=255)

    plt.subplot(2, print_amount, i+1+print_amount)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(result_images_test[i], cmap=plt.cm.gray, vmin=0, vmax=255)

plt.savefig('demo.png')


# --------------------------------------------------------------------------------------
# Test vector/scalar mult

if not __debug__:

    tic = time.process_time()
    neural_network._layer_forward(batches_train, 'in')
    toc = time.process_time()
    print("Input layer VECTOR \n ----- Computation time = " +
          str(1000*(toc - tic)) + "ms")

    tic = time.process_time()
    a = scalar_mult(batches_train, neural_network.w_in.T) + neural_network.b_in
    z = relu(a)
    toc = time.process_time()
    print("Input layer SCALAR \n ----- Computation time = " +
          str(1000*(toc - tic)) + "ms")

# Input layer VECTOR
#  ----- Computation time = 11.368929000000083ms
# Input layer SCALAR
#  ----- Computation time = 14744.943132000002ms
