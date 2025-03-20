import numpy as np

class capaDensa:
    def __init__(self, entradas: int, neuronas: int, l2_lambda=0.01):
        self.pesos = np.random.randn(entradas, neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))
        self.l2_lambda = l2_lambda  

        self.weight_momentums = np.zeros_like(self.pesos)
        self.weight_cache = np.zeros_like(self.pesos)
        self.bias_momentums = np.zeros_like(self.sesgos)
        self.bias_cache = np.zeros_like(self.sesgos)

    def forward(self, datos: np.array):
        self.entrada = datos  
        self.salida = np.dot(datos, self.pesos) + self.sesgos  

    def backward(self, gradiente_salida: np.array):
        self.gradiente_pesos = np.dot(self.entrada.T, gradiente_salida) + (self.l2_lambda * self.pesos)
        self.gradiente_sesgos = np.sum(gradiente_salida, axis=0, keepdims=True) 
        gradiente_entrada = np.dot(gradiente_salida, self.pesos.T) 
        return gradiente_entrada

    def update(self, optimizador):
        optimizador.update_params(self)

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
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        return exp / (sum_exp + 1e-9) 

    def backward(self, grad_output, outputs):
        return grad_output  

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / batch_size 
        return loss

    def backward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        return (y_pred - y_true) / batch_size  

class one_hot:
    @staticmethod
    def one_hot_encode(labels: np.array, num_classes: int):
        return np.eye(num_classes)[labels]
