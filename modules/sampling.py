from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer

def sampling(latent_dim):
    
    def sampling_inner(args):
        """sample from encoded embedding
        @param args (tuple): tuple of z_min and z_log_var
        @return: normalized variables
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    return sampling_inner