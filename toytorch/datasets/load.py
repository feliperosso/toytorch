# Load Packages
import toytorch, torch
import importlib, pickle, os

# - Load .txt data -
def load_txt_data(file_name:str):
    return importlib.resources.read_text(toytorch.datasets, file_name)
    
# - Load CIFAR10 data -
def load_CIFAR10_data():
    """ Loads the CIFAR10 dataset, properly divided into train, validation and test. 
        X_train: (40k, in_channels=3, image_size=(32, 32))
        Y_train: (40k) (Integers from 0-9 corresponding to classes)

        X_validation: (10k, in_channels=3, image_size=(32, 32))
        Y_validation: (10k) (Integers from 0-9 corresponding to classes)

        X_test: (10k, in_channels=3, image_size=(32, 32))
        Y_test: (10k) (Integers from 0-9 corresponding to classes)

        dictionary: Translates 0-9 int to classess
    """
    # Locate folder path
    toytorch_path = os.path.dirname(toytorch.__file__)
    folder_path = toytorch_path + '/datasets/CIFAR10Data'

    # Import the raw data files
    def unpickle(file_path):
        with open(file_path, 'rb') as file:
            dict = pickle.load(file, encoding='latin1')
        return dict

    batch1 = unpickle(folder_path + "/data_batch_1")
    batch2 = unpickle(folder_path + "/data_batch_2")
    batch3 = unpickle(folder_path + "/data_batch_3")
    batch4 = unpickle(folder_path + "/data_batch_4")
    batch5 = unpickle(folder_path + "/data_batch_5")

    test_batch = unpickle(folder_path + "/test_batch")

    # - RGB IMAGES -
    n_max = 255
    def to_torch_and_reshape(batch_data):
        n = len(batch_data)
        return torch.reshape(torch.from_numpy(batch_data), (n, 3, 32, 32))

    # Training Data
    X_train = to_torch_and_reshape(batch1["data"])
    X_train = torch.cat((X_train, to_torch_and_reshape(batch2["data"])), dim=0)
    X_train = torch.cat((X_train, to_torch_and_reshape(batch3["data"])), dim=0)
    X_train = torch.cat((X_train, to_torch_and_reshape(batch4["data"])), dim=0)/n_max

    # Validation and test Data
    X_validation = to_torch_and_reshape(batch5["data"])/n_max
    X_test = to_torch_and_reshape(test_batch["data"])/n_max


    # - LABELS -
    # For validation and test data the labels are just scalars
    Y_validation = torch.tensor(batch5["labels"])
    Y_test = torch.tensor(test_batch["labels"])

    # For training data the labels are vectorized
    Y_train = torch.tensor(batch1["labels"])
    Y_train = torch.cat((Y_train, torch.tensor(batch2["labels"])), dim=0)
    Y_train = torch.cat((Y_train, torch.tensor(batch3["labels"])), dim=0)
    Y_train = torch.cat((Y_train, torch.tensor(batch4["labels"])), dim=0)

    #Y_train = torch.zeros(len(X_train), 10) # This applies one_hot encoding to Y_train
    #for k, num in enumerate(Y_raw):
    #    Y_train[k, num] = 1

    # Create dictionary to translate the numbers to the categories
    batches_meta = unpickle(folder_path + "/batches.meta")
    dictionary = dict()
    for k, label_names in enumerate(batches_meta["label_names"]):
        dictionary[k] = label_names
    
    return (X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test), dictionary


# - EXTRA: For plotting the images with the corresponding labels -
import matplotlib.pyplot as plt

def plot_CIFAR10_data(X, Y, dictionary):
    image = X
    label = dictionary[Y.item()]

    image = torch.transpose(image, dim0=0, dim1=2)
    image = torch.transpose(image, dim0=0, dim1=1)

    plt.title(label)
    plt.imshow(image)

#plot_data(X_validation[30], Y_validation[30])