import numpy as np
from utils import load_and_prepare_data
from sklearn.metrics import accuracy_score

class LDA:
    def __init__(self):
        self.class_means = None
        self.shared_covariance = None

    def fit(self, X_train, y_train):
        # Number of classes
        num_classes = len(np.unique(y_train))
        
        # Initialize arrays to store class means and covariance matrices
        class_means = []
        class_covariances = []

        # Loop through each class
        for c in range(num_classes):
            # Select data points belonging to class c
            class_c_data = X_train[y_train == c]

            # Ensure class_c_data is a 2D array
            class_c_data_2d = class_c_data.reshape(class_c_data.shape[0], -1)

            # Calculate class mean
            class_mean = np.mean(class_c_data_2d, axis=0)
            class_means.append(class_mean)

            # Calculate class covariance matrix
            class_covariance = np.cov(class_c_data_2d, rowvar=False)
            class_covariances.append(class_covariance)

        # Convert lists to numpy arrays
        self.class_means = np.array(class_means)
        self.shared_covariance = np.mean(class_covariances, axis=0)

    def predict(self, X_test):
        # Flatten the input images to 1D arrays
        X_test_1d = X_test.reshape(X_test.shape[0], -1)

        # Calculate Mahalanobis distances to each class using matrix operations
        mahalanobis_distances = np.zeros((X_test.shape[0], self.class_means.shape[0]))

        for i, class_mean in enumerate(self.class_means):
            diff = X_test_1d - class_mean
            mahalanobis_distances[:, i] = np.sqrt(np.sum(np.dot(diff, np.linalg.inv(self.shared_covariance)) * diff, axis=1))

        # Predict the class with the minimum Mahalanobis distance
        predictions = np.argmin(mahalanobis_distances, axis=1)

        return predictions

print("****************************************")
print("*        LDA Solution        *")
print("****************************************")

print("Loading data...", end="")
X_train, y_train, X_test, y_test = load_and_prepare_data()
print("done.")

print("Fitting LDA model...", end="")
lda_model = LDA()
lda_model.fit(X_train, y_train)
print("done.")

# Make predictions on the test data
predictions = lda_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on the test data: {accuracy:.2%}")
