from pathlib import Path
import sys
import MnistDataset as MnistDs
import matplotlib.pyplot as plt

def displayImage(image, title=None):
    """
    Displays a single MNIST image.

    Parameters:
    - image: 2D array representing the image.
    - title: Optional title for the image.
    """
    image2 = image.reshape(28, 28)
    plt.imshow(image2, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')  # Hide axis ticks
    plt.show()

def checkFileExists(root_dir: str, files: list[str]) -> bool:
    file_exists = True
    root_dir_path = Path(root_dir)

    for file in files:
        filepath = root_dir_path / Path(file)
        if not filepath.exists() or not filepath.is_file():
            file_exists = False
            print(f"File '{filepath}' doesn't exits")

    return file_exists

test_images_file = "t10k-images-idx3-ubyte"
test_labels_file = "t10k-labels-idx1-ubyte"
train_images_file = "train-images-idx3-ubyte"
train_labels_file = "train-labels-idx1-ubyte"

dataset_files = [test_images_file, test_labels_file, train_images_file, train_labels_file]

if not checkFileExists("dataset", dataset_files):
    sys.exit(1)

ds_folder_path = Path("dataset")

test_images_path = ds_folder_path / "t10k-images-idx3-ubyte"
test_labels_path = ds_folder_path / "t10k-labels-idx1-ubyte"
train_images_path = ds_folder_path / "train-images-idx3-ubyte"
train_labels_path = ds_folder_path / "train-labels-idx1-ubyte"

#mnist_test = MnistDs.MnistDataset()
mnist_train = MnistDs.MnistDataset()

#mnist_test.load(test_images_path, test_labels_path)
mnist_train.load(train_images_path, train_labels_path)

print(mnist_train.images.shape)
print(mnist_train.labels.shape)

displayImage(mnist_train.images[0], "MNIST Image 1")
displayImage(mnist_train.images[1], "MNIST Image 2")

print(f"Labels: {mnist_train.labels[0]}, {mnist_train.labels[1]}")

for i in range(2):
    print(mnist_train.images[i]) 
    print("\n")