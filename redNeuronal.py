import numpy as np
from pathlib import Path
import sys
import MnistDataset as MnistDs
import capaDensa as cp

# Función para verificar si los archivos existen
def checkFileExists(root_dir: str, files: list[str]) -> bool:
    root_dir_path = Path(root_dir)
    for file in files:
        filepath = root_dir_path / file
        if not filepath.exists() or not filepath.is_file():
            print(f"File '{filepath}' doesn't exist")
            return False
    return True

# Configuración del dataset
dataset_path = "dataset"
train_images_file = "train-images-idx3-ubyte"
train_labels_file = "train-labels-idx1-ubyte"
dataset_files = [train_images_file, train_labels_file]

if not checkFileExists(dataset_path, dataset_files):
    sys.exit(1)

ds_folder_path = Path(dataset_path)
train_images_path = ds_folder_path / train_images_file
train_labels_path = ds_folder_path / train_labels_file
mnist_train = MnistDs.MnistDataset()
mnist_train.load(train_images_path, train_labels_path)

# Función para calcular precisión
def calc_precision(y_pred, y_true):
    predicciones = np.argmax(y_pred, axis=1)
    etiquetas = np.argmax(y_true, axis=1)
    return np.mean(predicciones == etiquetas)

# Hiperparámetros
batch_size = 64
num_epochs = 50
tasa_aprendizaje = 0.1
num_classes = 10

# Capas de la red
capa1 = cp.capaDensa(784, 128)
relu1 = cp.ReLU()
capa_salida = cp.capaDensa(128, 10)
softmax = cp.Softmax()
cross_entropy = cp.CrossEntropyLoss()

# Listas para almacenar métricas
perdidas_por_epoca = []
precisiones_por_epoca = []

# Bucle de entrenamiento
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    perdida_epoch = 0
    precision_epoch = 0
    num_batches = 0

    for i in range(0, len(mnist_train.images), batch_size):
        imagenes_lote = mnist_train.images[i:i + batch_size]
        etiquetas_lote = mnist_train.labels[i:i + batch_size]
        one_hot_labels = cp.one_hot.one_hot_encode(etiquetas_lote, num_classes)

        # Forward Pass
        capa1.forward(imagenes_lote)
        relu1.forward(capa1.salida)
        capa_salida.forward(relu1.salida)
        y_pred = softmax.forward(capa_salida.salida)

        # Cálculo de pérdida y precisión
        perdida = cross_entropy.forward(y_pred, one_hot_labels)
        precision = calc_precision(y_pred, one_hot_labels)
        perdida_epoch += perdida
        precision_epoch += precision
        num_batches += 1

        # Backward Pass con la simplificación de softmax y cross-entropy
        gradiente_loss = cross_entropy.backward(y_pred, one_hot_labels)
        gradiente_capa_salida = capa_salida.backward(gradiente_loss)  # No se necesita backward de Softmax
        gradiente_relu = relu1.backward(gradiente_capa_salida)
        gradiente_capa1 = capa1.backward(gradiente_relu)

        # Actualización de pesos
        capa1.update(tasa_aprendizaje)
        capa_salida.update(tasa_aprendizaje)

    # Promedios de pérdida y precisión por época
    perdida_promedio = perdida_epoch / num_batches
    precision_promedio = precision_epoch / num_batches

    perdidas_por_epoca.append(perdida_promedio)
    precisiones_por_epoca.append(precision_promedio)

    print(f"Pérdida: {perdida_promedio:.4f}, Precisión: {precision_promedio:.4f}")
