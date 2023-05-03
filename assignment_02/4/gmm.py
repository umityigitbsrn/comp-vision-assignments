import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixtureModel:
    def __init__(self, num_of_clusters, num_of_iterations=100, reshape_to=(512, 512)):
        self.__num_of_clusters = num_of_clusters
        self.__num_of_iterations = num_of_iterations

        self.__width, self.__height = None, None
        self.__magnitude, self.__weight_arr = None, None
        self.__mean_arr, self.__cov_arr = None, None
        self.__reshape_result = reshape_to

    def __expectation_step(self, X):
        # mle using preselected mean var and magnitude values
        likelihood = np.zeros((self.__width, self.__num_of_clusters))
        for i in range(self.__num_of_clusters):
            likelihood[:, i] = multivariate_normal.pdf(X, mean=self.__mean_arr[i], cov=self.__cov_arr[i],
                                                       allow_singular=True)
        self.__weight_arr = likelihood * self.__magnitude / np.sum(self.__weight_arr, axis=1)[:, np.newaxis]

    def __maximization_step(self, X):
        for i in range(self.__num_of_clusters):
            weight_sum = np.sum(self.__weight_arr[:, i])
            # update params depending on the cluster
            self.__mean_arr[i] = np.sum(X * self.__weight_arr[:, i][:, np.newaxis], axis=0) / weight_sum
            self.__cov_arr[i] = np.cov(X.T, aweights=self.__weight_arr[:, i], bias=True)
            self.__magnitude[i] = weight_sum / self.__width

    def __forward(self, X):
        # calculating probabilities for samples
        likelihood = np.zeros((X.shape[0], self.__num_of_clusters))
        for i in range(self.__num_of_clusters):
            likelihood[:, i] = multivariate_normal.pdf(X, mean=self.__mean_arr[i], cov=self.__cov_arr[i])
        upper = likelihood * self.__magnitude
        lower = np.sum(upper, axis=1)[:, np.newaxis]
        result = upper / lower
        return result

    def train(self, X):
        # setting parameters for training
        self.__width, self.__height = X.shape
        self.__magnitude = np.ones(self.__num_of_clusters) / self.__num_of_clusters
        self.__weight_arr = np.ones(X.shape) / self.__num_of_clusters

        # random initialization
        indices = np.random.choice(self.__width, self.__num_of_clusters, replace=False)
        self.__mean_arr = X[indices]
        self.__cov_arr = [np.cov(X.T) for _ in range(self.__num_of_clusters)]

        # training with expectation maximization
        for iteration in range(self.__num_of_iterations):
            self.__expectation_step(X)
            self.__maximization_step(X)

    def test(self, X):
        # directly generate segmented image and returns it
        clusters = np.argmax(self.__forward(X), axis=1)
        color_codes = [(255 // 10) * (x + 1) for x in range(self.__num_of_clusters)]
        result = np.zeros(clusters.shape, dtype=int)
        for idx, cluster in enumerate(clusters):
            result[idx] = color_codes[cluster]
        result = result.reshape(self.__reshape_result)
        result = np.stack([result, result, result], axis=-1)
        return result
