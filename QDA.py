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
        unique_classes = np.unique(y_train)
        num_classes = len(unique_classes)
        num_features = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]  # Account for multiple channels

        class_means = np.zeros((num_classes, num_features))
        class_covariances = np.zeros((num_classes, num_features, num_features))
        class_priors = np.zeros(num_classes)

        for c in unique_classes:
            class_c_data = X_train[y_train == c].reshape(-1, num_features)

            class_means[c] = np.mean(class_c_data, axis=0)
            class_covariances[c] = np.cov(class_c_data, rowvar=False)
            class_priors[c] = len(class_c_data) / len(X_train)

        self.class_means = class_means.reshape(num_classes, -1)  # Reshape to match the number of features
        self.class_covariances = class_covariances
        self.class_priors = class_priors
    def predict(self, X_test):
        num_classes = len(self.class_means)
        num_samples, height, width, channels = X_test.shape
        num_features = height * width * channels

    # Reshape and flatten test data to 2D array
        X_test_2d = X_test.reshape(num_samples, num_features)

    # Calculate Mahalanobis distances for all samples and classes
        mahalanobis_distances = np.zeros((num_samples, num_classes))
        for c in range(num_classes):
            diff = X_test_2d - self.class_means[c]
            inv_covariance = np.linalg.inv(self.class_covariances[c])
            mahalanobis_distances[:, c] = np.sum(diff @ inv_covariance * diff, axis=1)

    # Calculate QDA scores
        qda_scores = -0.5 * mahalanobis_distances - 0.5 * np.array([np.linalg.slogdet(cov)[1] for cov in self.class_covariances]) + np.log(self.class_priors)

    # Predict the class with the maximum QDA score
        predictions = np.argmax(qda_scores, axis=1)

        return predictions
    
# Load CIFAR-10 data
X_train, y_train, X_test, y_test = load_and_prepare_data()

# Instantiate QDA model
qda_model = QDA()

# Fit the model to the training data
qda_model.fit(X_train, y_train)

# Make predictions on the test data
predictions = qda_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

'''X_train, y_train, X_test, y_test = load_and_prepare_data()

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
print(f'Accuracy: {accuracy}')'''
