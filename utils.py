import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


def load_data():
    with open('datasets/train.pkl', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    with open('datasets/val.pkl', 'rb') as f:
        X_val, y_val = pickle.load(f)
    
    with open('datasets/test.pkl', 'rb') as f:
        X_test, y_test = pickle.load(f)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def model_info(model):
    model.summary()
    plot_model(model, to_file=f'model_plots/{model.name}.png', show_shapes=True)
    
    
def plot_history(history, name=None):
    train_history = history.history['loss']
    validation_history = history.history['val_loss']
    fig = plt.figure(figsize=(8, 8))
    plt.title('Loss History')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS Function')
    plt.plot(train_history, 'red', label='train')
    plt.plot(validation_history, 'blue', label='validation')
    plt.legend()
    if name:
        fig.savefig(f"history_plots/{name}_loss_history.png")
    else:
        fig.savefig(f"history_plots/{history.model.name}_loss_history.png")

    train_history = history.history['accuracy']
    validation_history = history.history['val_accuracy']
    fig = plt.figure(figsize=(8, 8))
    plt.title('Accuracy History')
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(train_history, 'red', label='train')
    plt.plot(validation_history, 'blue', label='validation')
    plt.legend()
    if name:
        fig.savefig(f"history_plots/{name}_loss_history.png")
    else:
        fig.savefig(f'history_plots/{history.model.name}_accuracy_history.png')
    
