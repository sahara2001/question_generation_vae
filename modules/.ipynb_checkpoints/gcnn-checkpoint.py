# GCNN layer
import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer

# no flow control
class GCNN(Layer): # 定义GCNN层，结合残差
    def __init__(self, output_dim=None, residual=False, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.residual = residual
    def build(self, input_shape):
        if self.output_dim == None:
            self.output_dim = input_shape[-1]
        self.kernel = self.add_weight(name='gcnn_kernel',
                                     shape=(3, input_shape[-1],
                                            self.output_dim * 2),
                                     initializer='glorot_uniform',
                                     trainable=True)
        # support mask
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return input_mask

    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            # mask (batch, x_dim, time)
            mask = K.repeat(mask, x.shape[-1])
            # mask (batch, time, x_dim)
            mask = tf.transpose(mask, [0,2,1])
            # to make the masked values in x be equal to zero
            x = x * mask
        _ = K.conv1d(x, self.kernel, padding='same')
        _ = _[:,:,:self.output_dim] * K.sigmoid(_[:,:,self.output_dim:])
        if self.residual:
            return _ + x
        else:
            return _
        
class ResCNN(Layer): # 定义CNN层，结合残差
    def __init__(self, output_dim=None, residual=False, **kwargs):
        super(ResCNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.residual = residual
        self.beta = 1.702
    def build(self, input_shape):
        if self.output_dim == None:
            self.output_dim = input_shape[-1]
        self.kernel = self.add_weight(name='gcnn_kernel',
                                     shape=(3, input_shape[-1],
                                            self.output_dim * 2),
                                     initializer='glorot_uniform',
                                     trainable=True)
        self.bm1 = BatchNormalization()
        self.bm2 = BatchNormalization()
        
              # support mask
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            # mask (batch, x_dim, time)
            mask = K.repeat(mask, x.shape[-1])
            # mask (batch, time, x_dim)
            mask = tf.transpose(mask, [0,2,1])
            x = x * mask
        
        x = self.bm1(x)
        _ = K.conv1d(x, self.kernel, padding='same')
        _ = K.relu(_[:,:,:self.output_dim] * K.sigmoid(_[:,:,self.output_dim:]))
        _ = self.bm2(_)
#         x * K.sigmoid(beta * x)
        if self.residual:
            return _ + x
        else:
            return _