import numpy as np


class myNeuralNetwork(object):
    def __init__(self, n_in, n_layer1, n_layer2, n_out, learning_rate=0.005):
        self.learning_rate = learning_rate
        self.n_in = n_in
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.n_out = n_out
        # Initialize random weights
        np.random.seed(100)
        self.weights_1 = np.random.randn(5, 2)
        self.weights_2 = np.random.randn(5, 5)
        self.weights_3 = np.random.randn(5, 1)
        # activations
        self.a1 = []
        self.a2 = []
        self.a3 = []
        # delta gradients
        self.z1 = []
        self.z2 = []
        self.z3 = []
        # weight gradients
        self.w1 = []
        self.w2 = []
        self.w3 = []
        # saved weights from training data
        self.savedw1 = []
        self.savedw2 = []
        self.savedw3 = []
        # learning curves
        self.validation_loss = []
        self.train_loss = []

    def forward_propagation(self, x):
        y_hat = self.predict_proba(x)
        return y_hat

    # Cross-entropy loss function
    def compute_loss(self, X, y, v=False):
        # check for validation data
        if v == True:
            for i in range(len(self.savedw1)):
                w1 = self.savedw1[i]
                w2 = self.savedw2[i]
                w3 = self.savedw3[i]
                y_hat = self.predict_proba_v(X, w1, w2, w3)
                loss = -(np.sum(y * np.log(y_hat)))
                self.validation_loss.append(loss / float(y_hat.shape[0]))
        # check for training data
        else:
            y_hat = self.predict_proba(X)
            loss = -(np.sum(y * np.log(y_hat)))
            return loss / float(y_hat.shape[0])

    def backpropagate(self, x, y):
        y_hat = self.forward_propagation(x)

        z3 = y_hat - y
        w3 = np.dot(self.a2.T, z3)

        z2 = np.dot(z3, w3.T) * self.sigmoid_derivative(self.z2)
        w2 = np.dot(self.a1.T, z2)

        z1 = np.dot(z2, w2.T) * self.sigmoid_derivative(self.z1)
        w1 = np.dot(z1.T, x)

        gradients = {"w1": w1, "w2": w2, "w3": w3}

        return gradients

    def stochastic_gradient_descent_step(self, gradients):
        dw1 = gradients["w1"]
        dw2 = gradients["w2"]
        dw3 = gradients["w3"]

        new_w1 = self.learning_rate * dw1
        new_w2 = self.learning_rate * dw2
        new_w3 = self.learning_rate * dw3

        n_w1 = self.weights_1 - new_w1
        n_w2 = self.weights_2 - new_w2
        n_w3 = self.weights_3 - new_w3

        self.weights_1 = n_w1
        self.weights_2 = n_w2
        self.weights_3 = n_w3

    def fit(self, X, y, max_epochs=5000, get_validation_loss=False):
        # check for validation data
        if get_validation_loss == True:
            self.compute_loss(X, y, v=True)
        # check for training data
        else:
            full = np.append(X, y, axis=-1)
            for _ in range(max_epochs):
                self.savedw1.append(self.weights_1)
                self.savedw2.append(self.weights_2)
                self.savedw3.append(self.weights_3)
                loss = self.compute_loss(X, y)
                self.train_loss.append(loss)
                np.random.shuffle(full)
                for i in full:
                    single_x = i[0:2].reshape(1, -1)
                    single_y = i[2:].reshape(1, -1)
                    gradients = self.backpropagate(single_x, single_y)
                    self.stochastic_gradient_descent_step(gradients)

    def predict_proba(self, X):
        self.a1 = np.dot(X, self.weights_1.T)
        self.z1 = self.sigmoid(self.a1)
        self.a2 = np.dot(self.z1, self.weights_2)
        self.z2 = self.sigmoid(self.a2)
        self.a3 = np.dot(self.z2, self.weights_3)
        self.z3 = self.sigmoid(self.a3)
        y_hat = self.z3
        return y_hat

    # Additional function for calculating y_hat for
    # validation data on saved weights
    def predict_proba_v(self, X, w1, w2, w3):
        l1 = np.dot(X, w1.T)
        l2 = self.sigmoid(l1)
        l3 = np.dot(l2, w2)
        l4 = self.sigmoid(l3)
        l5 = np.dot(l4, w3)
        l6 = self.sigmoid(l5)
        y_hat = l6
        return y_hat

    def predict(self, X, decision_thresh=0.5):
        y_labels = []
        y_hat = self.predict_proba(X)
        for i in y_hat:
            if i < decision_thresh:
                y_labels.append(0)
            else:
                y_labels.append(1)
        return y_labels

    def sigmoid(self, X):
        z = 1 / (1 + np.exp(-X))
        return z

    def sigmoid_derivative(self, X):
        z = self.sigmoid(X)
        z_hat = z * (1 - z)
        return z_hat
