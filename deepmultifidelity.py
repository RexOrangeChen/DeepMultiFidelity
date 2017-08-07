import numpy as np
import keras
from keras import layers
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import random as rn
import scipy.optimize as opt

#reproducible setup
#os.environ['PYTHONHASHSEED'] = '0'
#np.random.seed(425)
#rn.seed(1235)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#tf.set_random_seed(123)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)



#realfunc and data setup
def inputx(n,scale):
    return scale * np.random.random(n)

def realfunc(x):
    return np.sin(np.pi * x)

#data parameter set up
global rho
global test
global x_low
global x_high
test= 0
n_high = 3
n_low = 10
rho = 4
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
    print(x)
    length_x, = K.int_shape(x)
    length_y, = K.int_shape(y)
    X = K.repeat_elements(x, length_y, 0)
    Y = K.repeat_elements(y, length_x, 0)
    #form the kernel
    X = K.reshape(X, (length_x, length_y))
    Y = K.transpose(K.reshape(Y, (length_y, length_x)))
    M = X - Y
    M = K.pow(M, 2)
    print(K.cast(K.pow(hyp[1],2),'float32'))
    ker = K.cast(K.pow(hyp[0],2),'float32') * K.exp(-K.cast(K.pow(hyp[1],2),'float32') * M)
    return ker

#gp setup
def gp(x, hyp, n_high, n_low):
    hyp = np.reshape(hyp,(1,7))
    rho = hyp[0,6]#reset up for convenience
    x_l = x[0:n_low]
    x_h = x[n_low:n_high + n_low]
    K_LL = kerne1(x_l, x_l, hyp[0,0:3]) + K.eye(n_low)* K.cast(K.pow(hyp[0,2],2),'float32')
    K_LH = rho*kerne1(x_l, x_h, hyp[0,0:3])
    K_HL = rho*kerne1(x_h, x_l, hyp[0,0:3])
    K_HH = K.cast(K.pow(rho, 2),'float32') * kerne1(x_h, x_h, hyp[0,0:3]) \
                + kerne1(x_h, x_h, hyp[0,3:5]) + K.eye(n_high)* K.cast(K.pow(hyp[0,5],2),'float32')
    k_up = K.concatenate([K_LL, K_LH], axis = -1)
    k_down = K.concatenate([K_HL, K_HH], axis = -1)
    k = K.concatenate([k_up, k_down], axis = 0)
    return k

#to lower the computational complexity
def kerne1_normal(x, y, hyp):
    length_x, = np.shape(x)
    length_y, = np.shape(y)
    X = np.repeat(x, length_y, 0)
    Y = np.repeat(y, length_x, 0)
    #form the kernel
    X = np.reshape(X, (length_x, length_y))
    Y = np.reshape(X, (length_y, length_x)).transpose()
    M = X - Y
    M = np.power(M, 2)
    ker = hyp[0]**2 * np.exp(-np.power(hyp[1],2) * M)
    return ker

def gp_normal(x, hyp, n_high, n_low):
    rho = hyp[0,6]#reset up for convenience
    x_l = x[0:n_low]
    x_h = x[n_low:n_high + n_low]
    K_LL = kerne1_normal(x_l, x_l, hyp[0,0:3]) + np.eye(n_low)* hyp[0,2]**2
    K_LH = rho*kerne1_normal(x_l, x_h, hyp[0,0:3])
    K_HL = rho*kerne1_normal(x_h, x_l, hyp[0,0:3])
    K_HH = rho**2 * kerne1_normal(x_h, x_h, hyp[0,0:3]) \
                + kerne1_normal(x_h, x_h, hyp[0,3:5]) + np.eye(n_high)* hyp[0,5]**2
    k_up = np.concatenate([K_LL, K_LH], axis = -1)
    k_down = np.concatenate([K_HL, K_HH], axis = -1)
    k = np.concatenate([k_up, k_down], axis = 0)
    return k

#define gplayer
class gplayer(Layer):

    def __init__(self, n_high, n_low):
        self.n_high = n_high
        self.n_low = n_low
        self.hyp = np.array([[1],[1],[1],[1],[1],[1],[1]], dtype = 'float64')
        super(gplayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',shape=(1,0),
                                      initializer='ones',trainable=False)
        self.built = True
        super(gplayer, self).build(input_shape)

    def call(self, x):
        print(23334555)
        gpx = K.flatten(x[0])
        g = Model(inputs=a[0], outputs=h[0])
        def likelihood(hyp):
                gx = g.predict(np.concatenate([x_low, y_high]))
                print(gx)
                hyp = np.reshape(hyp,(1,7))
                k = gp_normal(np.ndarray.flatten(gx), hyp, self.n_high, self.n_low)
                yk = np.dot(y.transpose(), np.linalg.inv(k))
                k = np.dot(yk,y) + np.log(np.linalg.det(k))\
                        + (self.n_high + self.n_low)* np.log(2*np.pi)
                print(6)
                return k
        hyp = self.hyp
        #hyp = opt.fmin(likelihood, hyp)
        self.hyp = hyp
        k = gp(gpx, self.hyp, self.n_high, self.n_low)
        yk = K.dot(K.transpose(K.constant(y)), tf.matrix_inverse(k))
        k = K.dot(yk,K.constant(y)) + K.log(tf.matrix_determinant(k))\
                + (self.n_high + self.n_low)* K.log(K.constant(2*np.pi))
        return k

    def compute_output_shape(self, input_shape):
        return  (1, 1)

    def gethyp(self):
        return self.hyp

def er(y_true, y_pred):
    return (y_true + y_pred)

#build neural network
x = []
for i in range(layers):
    x.append(Dense(10, activation='sigmoid'))

a = []
for i in range(n_high + n_low):
    a.append(Input(shape = (1,)))
d = Dense(1)

h = []
for i in range(n_high + n_low):
    h.append(x[0](a[i]))
    for j in range(layers-1):
        h[i] = x[j+1](h[i])
    h[i] = d(h[i])


merged_vector =  keras.layers.concatenate(h, axis=-1)
gplayers = gplayer(n_high = n_high, n_low = n_low)
predictions = gplayers(merged_vector)

model = Model(inputs=a, outputs=predictions)
test = 1
#train
model.compile(optimizer='sgd',
              loss=er,
              metrics=['accuracy'])
print(np.split(np.concatenate([x_low, x_high], axis =0), n_high+n_low))
model.fit(np.split(np.concatenate([x_low, x_high], axis =0), n_high+n_low),np.array([0]), epochs=20)

hyp = gplayers.gethyp()
#plot
m = 20
x_pred = np.linspace(-scale, scale, m)
#g


g = Model(inputs=a[0], outputs=h[0])
gx_pred = g.predict(x_pred)
print(gx_pred)
gx = g.predict(np.concatenate([x_low, y_high]))
hyp = np.reshape(hyp,(1,7))
ker1 = gp_normal(np.ndarray.flatten(gx), hyp, n_high, n_low)
ker1 = np.linalg.inv(ker1)

y_pred = []
for n in gx_pred:
    kstar = K.eval(kerne1(K.constant(n),
                    K.constant(np.concatenate([x_low, x_high])), hyp))
    answ = np.dot(np.dot(kstar, ker1), y)
    print(n)
    print(answ)
    y_pred.append(answ)

y_pred = np.reshape(y_pred,(m,))
print(x_low)
print(y_pred)
import matplotlib.pyplot as plt
plt.plot(x_low, y_low)
plt.show()
