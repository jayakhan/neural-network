"""Load libraries"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nn


def test():
    # Load synthetic dataset
    train = datasets.make_moons(
        n_samples=600, shuffle=True, noise=0.20, random_state=20
    )

    # Split into train, validation, and test data
    X_trainv_m, X_test_m, y_trainv_m, y_test_m = train_test_split(
        train[0], train[1], test_size=0.166, random_state=0
    )
    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(
        X_trainv_m, y_trainv_m, test_size=0.20, random_state=0
    )

    y_train_m = y_train_m.reshape(-1, 1)
    y_val_m = y_val_m.reshape(-1, 1)
    y_test_m = y_test_m.reshape(-1, 1)

    """2. Fit train data to neural network"""
    # Fit and calculate train and validation loss
    # number of nodes is changed during hyperparamtere tuning
    nnet = nn.myNeuralNetwork(2, 10, 10, 1)
    nnet.fit(X_train_m, y_train_m)

    # Calculate validation loss
    nnet.fit(X_val_m, y_val_m, get_validation_loss=True)

    """3. Code to create learning curves"""
    # Function to visualization learning curve
    plt.plot(nnet.train_loss, color="blue", label="Training Error")
    plt.plot(nnet.validation_loss, color="red", label="Validation Error")

    plt.xlabel("Epoch", fontdict={"fontsize": 15})
    plt.ylabel("Loss", fontdict={"fontsize": 15})
    plt.grid(axis="y", linestyle="-")
    plt.title(
        "Comparison between Training and Validation Learning Curve",
        fontdict={"fontsize": 18},
    )
    plt.legend(bbox_to_anchor=(1.15, 0.6))
    plt.show()


if __name__ == "__main__":
    test()
