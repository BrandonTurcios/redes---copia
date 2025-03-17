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
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        return exp / (sum_exp + 1e-9)

    def backward(self, grad_output, outputs):
        return grad_output * outputs * (1 - outputs)

class CrossEntropyLoss:
     def forward(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

     def backward(self, y_true, y_pred):
        return y_pred-y_true


class one_hot:
    def one_hot_encode(labels: np.array, num_classes: int):
        return np.eye(num_classes)[labels]  