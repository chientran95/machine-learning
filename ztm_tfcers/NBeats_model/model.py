import tensorflow as tf
from tensorflow.keras import layers


class NBeatsBlock(layers.Layer):
    def __init__(self, input_size: int, theta_size: int, horizon: int,
                 n_neurons: int, n_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        
        self.hidden = [layers.Dense(self.n_neurons, activation='relu') for _ in range(self.n_layers)]
        self.theta_layer = layers.Dense(self.theta_size, activation='linear', name='theta')
    
    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast
