import numpy as np
import cv2
from joblib import Parallel, delayed
from utils import load_and_prepare_data
from sklearn.metrics import accuracy_score

class QDA:
    def __init__(self):
        self.class_means = None
        self.class_covariances = None
        self.class_priors = None

    def fit(self, X_train, y_train):
        num_classes = len(np.unique(y_train))

        class_means = []
        class_covariances = []
        class_priors = []

        for c in range(num_classes):
            class_c_data = X_train[y_train == c]
            class_c_data_2d = class_c_data.reshape(class_c_data.shape[0], -1)

            class_mean = np.mean(class_c_data_2d, axis=0)
            class_means.append(class_mean)

            class_covariance = np.cov(class_c_data_2d, rowvar=False)
            class_covariances.append(class_covariance)

            class_prior = len(class_c_data) / len(X_train)
            class_priors.append(class_prior)

        self.class_means = np.array(class_means)
        self.class_covariances = np.array(class_covariances)
        self.class_priors = np.array(class_priors)

    def predict(self, X_test):
        def calculate_qda_score(sample_1d, class_mean, class_covariance, class_prior):
            mahalanobis_distance = np.dot(np.dot((sample_1d - class_mean).T, np.linalg.inv(class_covariance)),
                                          (sample_1d - class_mean))
            qda_score = -0.5 * mahalanobis_distance - 0.5 * np.linalg.slogdet(class_covariance)[1] + np.log(
                class_prior)
            return qda_score

        def predict_single_sample(sample):
            sample_1d = sample.flatten()
            qda_scores = [calculate_qda_score(sample_1d, self.class_means[i], self.class_covariances[i],
                                              self.class_priors[i]) for i in range(len(self.class_means))]
            return np.argmax(qda_scores)

        # Use joblib to parallelize the loop
        predictions = Parallel(n_jobs=-1)(delayed(predict_single_sample)(sample) for sample in X_test)

        return np.array(predictions)

# Load CIFAR-10 data
X_train, y_train, X_test, y_test = load_and_prepare_data()

# Resize images to 16x16
X_train_resized = np.array([cv2.resize(img, (16, 16)) for img in X_train])
X_test_resized = np.array([cv2.resize(img, (16, 16)) for img in X_test])

# Instantiate QDA model
qda_model = QDA()

# Fit the model to the training data
qda_model.fit(X_train_resized, y_train)

# Make predictions on the resized test data
predictions = qda_model.predict(X_test_resized)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
