import numpy as np

class LDA:
    def __init__(self):
        # Initialize any required variables
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
        # Initialize an array to store predicted labels
        predictions = []

        # Loop through each data point in the test set
        for sample in X_test:
            # Flatten the input image to a 1D array
            sample_1d = sample.flatten()

            # Calculate Mahalanobis distances to each class
            mahalanobis_distances = []
            for class_mean in self.class_means:
                # Calculate Mahalanobis distance using the shared covariance matrix
                mahalanobis_distance = np.sqrt(np.dot(np.dot((sample_1d - class_mean).T, np.linalg.inv(self.shared_covariance)), (sample_1d - class_mean)))
                mahalanobis_distances.append(mahalanobis_distance)

            # Predict the class with the minimum Mahalanobis distance
            predicted_class = np.argmin(mahalanobis_distances)
            predictions.append(predicted_class)

        # Convert list to numpy array
        return np.array(predictions)



# The rest of your code remains unchanged
from utils import load_and_prepare_data
from sklearn.metrics import accuracy_score

# Load and prepare the data
X_train, y_train, X_test, y_test = load_and_prepare_data()

# Initialize and fit LDA model
lda_model = LDA()
lda_model.fit(X_train, y_train)

# Make predictions on the test data
predictions = lda_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on the test data: {accuracy:.2%}")
