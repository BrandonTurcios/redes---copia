import numpy as np

class Optimizer_Adam:
    def __init__(self, learning_rate, decay, epsilon, beta_1, beta_2):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.pesos)
            layer.weight_cache = np.zeros_like(layer.pesos)
            layer.bias_momentums = np.zeros_like(layer.sesgos)
            layer.bias_cache = np.zeros_like(layer.sesgos)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.gradiente_pesos
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
                               (1 - self.beta_1) * layer.gradiente_sesgos

        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.gradiente_pesos ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.gradiente_sesgos ** 2

        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        layer.pesos -= self.current_learning_rate * weight_momentums_corrected / \
                       (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.sesgos -= self.current_learning_rate * bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
