import numpy as np

class capaDensa:
    def __init__(self, entradas: int, neuronas: int):
        self.pesos = np.random.randn(entradas, neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))

    def forward(self, datos: np.array):
        self.entrada = datos  
        self.salida = np.matmul(datos, self.pesos) + self.sesgos  

    def backward(self, gradiente_salida: np.array):
        self.gradiente_pesos = np.dot(self.entrada.T, gradiente_salida)  
        self.gradiente_sesgos = np.sum(gradiente_salida, axis=0, keepdims=True) 
        gradiente_entrada = np.dot(gradiente_salida, self.pesos.T) 
        return gradiente_entrada

    def update(self, tasa_aprendizaje: float):
        self.pesos -= tasa_aprendizaje * self.gradiente_pesos
        self.sesgos -= tasa_aprendizaje * self.gradiente_sesgos

class ReLU:
    def forward(self, x: np.array):
        self.entrada = x
        self.salida = np.maximum(0, x)

    def backward(self, gradiente_salida: np.array):
        gradiente_entrada = gradiente_salida.copy()
        gradiente_entrada[self.entrada <= 0] = 0  
        return gradiente_entrada
        
class Softmax:
    def forward(self, x: np.array):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))  # Evitar overflow
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        return exp / (sum_exp + 1e-9)  # Pequeño valor para estabilidad numérica

    def backward(self, grad_output, outputs):
        return grad_output  # La simplificación elimina la necesidad de calcular la jacobiana



class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / batch_size  # Evitar log(0)
        return loss

    def backward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        return (y_pred - y_true) / batch_size  # Derivada simplificada



class one_hot:
    def one_hot_encode(labels: np.array, num_classes: int):
        return np.eye(num_classes)[labels]  
import numpy as np

class capaDensa:
    def __init__(self, entradas: int, neuronas: int):
        self.pesos = np.random.randn(entradas, neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))

    def forward(self, datos: np.array):
        self.entrada = datos  
        self.salida = np.matmul(datos, self.pesos) + self.sesgos  

    def backward(self, gradiente_salida: np.array):
        self.gradiente_pesos = np.dot(self.entrada.T, gradiente_salida)  
        self.gradiente_sesgos = np.sum(gradiente_salida, axis=0, keepdims=True) 
        gradiente_entrada = np.dot(gradiente_salida, self.pesos.T) 
        return gradiente_entrada

    def update(self, tasa_aprendizaje: float):
        self.pesos -= tasa_aprendizaje * self.gradiente_pesos
        self.sesgos -= tasa_aprendizaje * self.gradiente_sesgos

class ReLU:
    def forward(self, x: np.array):
        self.entrada = x
        self.salida = np.maximum(0, x)

    def backward(self, gradiente_salida: np.array):
        gradiente_entrada = gradiente_salida.copy()
        gradiente_entrada[self.entrada <= 0] = 0  
        return gradiente_entrada
        
class Softmax:
    def forward(self, x: np.array):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))  # Evitar overflow
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        return exp / (sum_exp + 1e-9)  # Pequeño valor para estabilidad numérica

    def backward(self, grad_output, outputs):
        return grad_output  # La simplificación elimina la necesidad de calcular la jacobiana



class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / batch_size  # Evitar log(0)
        return loss

    def backward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        return (y_pred - y_true) / batch_size  # Derivada simplificada



class one_hot:
    def one_hot_encode(labels: np.array, num_classes: int):
        return np.eye(num_classes)[labels]  


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
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights -= self.current_learning_rate * weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.current_learning_rate * bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
