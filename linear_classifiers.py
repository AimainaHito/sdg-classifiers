#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt

from gaussian import n_gaussians


class LogRegression():

    def __init__(self, num_features, learning_rate=0.1):
        self._W = np.zeros(num_features)
        self._b = 0
        self._learning_rate = learning_rate


    def classify(self, sample):
        return 1 / (1 + (np.exp(-((self._W @ sample) + self._b))))


    def batch_classify(self, samples):
        return 1 / (1 + (np.exp(-((self._W @ samples.T) + self._b))))


    def gradient_descent(self, sample, label):
        error = label - self.classify(sample)
        self._W += self._learning_rate * (error * sample)
        self._b += self._learning_rate * error


    def batch_gradient_descent(self, batch, labels):
        errors = labels - self.batch_classify(batch)
        self._W += self._learning_rate * (errors.reshape(errors.size, 1) * batch).mean(0)
        self._b += self._learning_rate * errors.mean()


class SoftmaxRegression():

    def __init__(self, num_features, num_classes, learning_rate=0.1):
        self._W = np.zeros((num_classes, num_features))
        self._b = np.zeros(num_classes)
        self._learning_rate = learning_rate


    def classify(self, sample):
        return ((self._W @ sample) + self._b).argmax()


    def batch_classify(self, samples):
        return ((self._W @ samples.T).T + self._b).argmax(1)


    def _compute_classification_matrix(self, sample):
        class_weights = np.exp(((self._W @ sample) + self._b) - self._W.max())
        class_weights /= class_weights.sum()
        return class_weights


    def _batch_compute_classification_matrix(self, sample):
        class_weights = np.exp(((self._W @ sample.T).T + self._b) - self._W.max())
        class_weights = (class_weights.T / class_weights.sum(1)).T
        return class_weights


    def batch_gradient_descent(self, batch, labels):
        classification = self._batch_compute_classification_matrix(batch)
        label_matches = np.zeros((labels.size, self._W.shape[0]))
        for i, label in enumerate(labels):
            label_matches[i, label] = 1

        for i in range(self._W.shape[0]):
            self._W[i] += self._learning_rate * ((label_matches - classification).T[i].reshape((-1, 1)) * batch).mean(0)

        self._b += self._learning_rate * (label_matches - classification).mean(0)


    def gradient_descent(self, sample, label):
        classification = self._compute_classification_matrix(sample)
        #uses a one_hot vector for efficiently computing kroeninger's delta
        one_hot = np.eye(1, self._W.shape[0], label).ravel()

        self._W += self._learning_rate * (one_hot - classification).reshape(-1, 1) * sample.reshape(1, -1)
        self._b += self._learning_rate * one_hot - classification


def get_class(probability):
    return int(probability > .5)


def test_logistic_accuracy(model, samples, labels):
    correct = 0
    for sample_batch, label_batch in zip(np.split(samples, BATCH_SIZE), np.split(labels, BATCH_SIZE)):
        classes = map(get_class, model.batch_classify(sample_batch))
        correct += sum(int(c == label) for c, label in zip(classes, label_batch))

    return correct / len(labels)

def test_softmax_accuracy(model, samples, labels):
    correct = 0
    for sample_batch, label_batch in zip(np.split(samples, BATCH_SIZE), np.split(labels, BATCH_SIZE)):
        correct += sum(int(c == label) for c, label in zip(model.batch_classify(sample_batch), label_batch))

    return correct / len(labels2)


if __name__ == '__main__':
    BATCH_SIZE = 50
    MAX_ITERATIONS = 25
    LEARNING_RATE = 0.5

    #create training sets
    basic_samples, basic_labels = n_gaussians(np.array([[0.2, 0.8], [0.8, 0.2]]), np.array([0, 1]), 600)
    basic_samples2, basic_labels2 = n_gaussians(np.array([[0.2, 0.2], [1.0, 0.0], [1.0, 1.0]]), np.array([0, 1, 2]), 600)
    samples, labels = n_gaussians(np.array([[0.2, 0.8], [0.8, 0.1], [0.4, .6]]), np.array([0, 1, 0]), 600)
    samples2, labels2 = n_gaussians(np.array([[0.2, 0.8], [0.8, 0.1], [0.1, 0.3], [1, 0.9]]), np.array([0, 1, 2, 3]), 600)

    model = LogRegression(2, LEARNING_RATE)

    initial_accuracy = test_logistic_accuracy(model, samples, labels)
    accuracy_per_iteration = [initial_accuracy]
    for i in range(MAX_ITERATIONS):
        #Create batches
        for sample_batch, label_batch in zip(np.split(samples, BATCH_SIZE), np.split(labels, BATCH_SIZE)):
            model.batch_gradient_descent(sample_batch, label_batch)

        accuracy = test_logistic_accuracy(model, samples, labels)
        accuracy_per_iteration.append(accuracy)

        if accuracy >= 1: #Stop iterating when the model converges
            break


    #Create a SoftmaxRegression models for 4 class classification
    model2 = SoftmaxRegression(2, 4, LEARNING_RATE)

    initial_accuracy = test_softmax_accuracy(model2, samples2, labels2)

    softmax_accuracy_per_iteration = [initial_accuracy]
    for i in range(MAX_ITERATIONS):
        #Create batches
        for sample, label in zip(np.split(samples2, BATCH_SIZE), np.split(labels2, BATCH_SIZE)):
            model2.batch_gradient_descent(sample, label)

        accuracy = test_softmax_accuracy(model2, samples2, labels2)
        softmax_accuracy_per_iteration.append(accuracy)
        if accuracy >= 1:
            break


    #Plotting results

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].set_title("Logistic Regression Learning")
    axis[0, 0].plot(accuracy_per_iteration)
    axis[0, 0].set_ylabel("Accuracy")
    text = "Learning rate: {:.2f}\nFinal accuracy: {:.2f}%\nGlobal Maximum: {:.2f}%".format(
        LEARNING_RATE, accuracy_per_iteration[-1] * 100, max(accuracy_per_iteration) * 100
    )
    axis[0, 0].text(0.25, 0.5, text, transform=axis[0, 0].transAxes, fontsize=14,
                    verticalalignment='top', bbox={'boxstyle' : 'square', 'facecolor' : 'white', 'alpha' : .8})

    class_1 = np.array([sample for sample, label in zip(samples, labels) if label == 0])
    class_2 = np.array([sample for sample, label in zip(samples, labels) if label == 1])

    axis[0, 1].set_title("Datapoints")

    #Decision boundary
    xx, yy = np.mgrid[-.4:1.2:.01, -.4:1.2:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.batch_classify(grid).reshape(xx.shape)
    axis[0, 1].contourf(xx, yy, probs, 50, cmap="RdBu",
                        vmin=0, vmax=1)

    axis[0, 1].scatter(class_2[:, 0], class_2[:, 1])
    axis[0, 1].scatter(class_1[:, 0], class_1[:, 1])

    axis[1, 0].set_title("Softmax Regression Learning")
    axis[1, 0].plot(softmax_accuracy_per_iteration)
    axis[1, 0].set_xlabel("Iterations")
    axis[1, 0].set_ylabel("Accuracy")
    text = "Learning rate: {:.2f}\nFinal accuracy: {:.2f}%\nGlobal Maximum: {:.2f}%".format(
        LEARNING_RATE, softmax_accuracy_per_iteration[-1] * 100, max(softmax_accuracy_per_iteration) * 100
    )
    axis[1, 0].text(0.25, 0.5, text, transform=axis[1, 0].transAxes, fontsize=14,
                    verticalalignment='top', bbox={'boxstyle' : 'square', 'facecolor' : 'white', 'alpha' : .8})


    classes_1 = np.array([sample for sample, label in zip(samples2, labels2) if label == 0])
    classes_2 = np.array([sample for sample, label in zip(samples2, labels2) if label == 1])
    classes_3 = np.array([sample for sample, label in zip(samples2, labels2) if label == 2])
    classes_4 = np.array([sample for sample, label in zip(samples2, labels2) if label == 3])

    #Decision boundaries
    xx, yy = np.mgrid[-.4:1.6:.01, -.4:1.6:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model2.batch_classify(grid).reshape(xx.shape)
    axis[1, 1].contourf(xx, yy, probs, 50, cmap="tab20b",
                        vmin=0, vmax=3)

    axis[1, 1].scatter(classes_1[:, 0], classes_1[:, 1])
    axis[1, 1].scatter(classes_2[:, 0], classes_2[:, 1])
    axis[1, 1].scatter(classes_3[:, 0], classes_3[:, 1])
    axis[1, 1].scatter(classes_4[:, 0], classes_4[:, 1])

    plt.show()
