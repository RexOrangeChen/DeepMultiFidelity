import numpy as np
import keras
from keras import layers
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping
import tensorflow as tf
import random as rn
import os

#reproducible setup
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(425)
rn.seed(1235)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(123)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#realfunc and data setup
def inputx(n,scale):
    return scale * np.random.random(n)

def realfunc(x):
    return np.sin(np.pi * x)

#data parameter set up
global rho

n_high = 3
n_low = 10
rho = 1.25
noise_L = 0
noise_H = 0
layers = 5
scale = 1

#initalize
x_high = inputx(n_high, scale)
x_high.sort()
x_low = inputx(n_low, scale)
x_low.sort()
y_high = realfunc(x_high) + noise_H * np.random.normal(size = n_high)
y_low = realfunc(x_low) / rho + noise_L * np.random.normal(size = n_low)
global y
y = np.concatenate([y_low, y_high], axis =0)
y = np.reshape(y,(n_high+n_low,1))

#kernel setup
def kerne1(x, y, hyp):
    length_x, = K.int_shape(x)
    length_y, = K.int_shape(y)

    X = K.repeat_elements(x, length_y, 0)
    Y = K.repeat_elements(y, length_x, 0)
    #form the kernel
    X = K.reshape(X, (length_x, length_y))
    Y = K.transpose(K.reshape(Y, (length_y, length_x)))
    M = X - Y
    M = K.pow(M, 2)
    ker = K.pow(hyp[0],2) * K.exp(-K.pow(hyp[1],2) * M)
    return ker

#gp setup
def gp(x, hyp, n_high, n_low):
    rho = hyp[0,6]#reset up for convenience
    x_l = x[0:n_low]
    x_h = x[n_low:n_high + n_low]
    K_LL = kerne1(x_l, x_l, hyp[0,0:3]) + K.eye(n_low)* K.pow(hyp[0,2],2)
    K_LH = rho*kerne1(x_l, x_h, hyp[0,0:3])
    K_HL = rho*kerne1(x_h, x_l, hyp[0,0:3])
    K_HH = K.pow(rho, 2) * kerne1(x_h, x_h, hyp[0,0:3]) \
                + kerne1(x_h, x_h, hyp[0,3:5]) + K.eye(n_high)* K.pow(hyp[0,5],2)
    k_up = K.concatenate([K_LL, K_LH], axis = -1)
    k_down = K.concatenate([K_HL, K_HH], axis = -1)
    k = K.concatenate([k_up, k_down], axis = 0)
    return k

#define gplayer
class gplayer(Layer):

    def __init__(self, n_high, n_low):
        self.n_high = n_high
        self.n_low = n_low
        super(gplayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',shape=(1,7),
                                      initializer='ones',trainable=True)
        self.built = True
        super(gplayer, self).build(input_shape)

    def call(self, x):
        gpx = K.flatten(x[0])
        k = gp(gpx, self.kernel, self.n_high, self.n_low)
        yk = K.dot(K.transpose(K.constant(y)), tf.matrix_inverse(k))
        print(K.int_shape(yk))
        k = K.dot(yk,K.constant(y)) + K.log(tf.matrix_determinant(k))\
                + (self.n_high + self.n_low)* K.log(K.constant(2*np.pi))
        return k

    def compute_output_shape(self, input_shape):
        return  (1, 1)


def er(y_true, y_pred):
    return (y_true + y_pred)

#build neural network
layer = []
for i in range(layers):
    layer.append(Dense(10, activation='sigmoid', kernel_initializer='ones'))

a = []
for i in range(n_high + n_low):
    a.append(Input(shape = (1,)))
dense = Dense(1)

h = []
for i in range(n_high + n_low):
    h.append(layer[0](a[i]))
    for j in range(layers-1):
        h[i] = layer[j+1](h[i])
    h[i] = dense(h[i])

merged_vector =  keras.layers.concatenate(h, axis=1)
gplayers = gplayer(n_high = n_high, n_low = n_low)
predictions = gplayers(merged_vector)

model = Model(inputs=a, outputs=predictions)

#train
model.compile(optimizer='adam',
              loss=er,
              metrics=['accuracy'])

model.fit(np.split(np.concatenate([x_low, x_high], axis =0),n_high+n_low),
                      np.array([0]), epochs=2000)

hyp1 = gplayers.get_weights()[0]
print(hyp1)

#plot
m = 20
x_pred = np.linspace(-scale, scale, m)
#g
g = Model(inputs=a[0], outputs=h[0])

gx_pred = g.predict(x_pred)

gx = K.constant(g.predict(np.concatenate([x_low, y_high])))
ker1 = K.eval(gp(K.flatten(gx), hyp1, n_high, n_low))

ker1 = np.linalg.inv(ker1)
hyp1 = hyp1.transpose()
print(gx_pred)
y_pred = []
for n in gx_pred:
    kstar = K.eval(kerne1(K.constant(n),
                    K.constant(np.concatenate([x_low, x_high])), hyp1))
    answ = np.dot(np.dot(kstar, ker1), y)

    y_pred.append(answ)

y_pred = np.reshape(y_pred,(m,))
print(y_pred)
import matplotlib.pyplot as plt
plt.plot(x_low, y_low)
plt.show()
