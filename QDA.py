import numpy as np

class QDA:
    def __init__(self):
        # Initialize any required variables
        self.class_means = None
        self.class_covariances = None
        self.class_priors = None

    def fit(self, X_train, y_train):
        # Number of classes
        num_classes = len(np.unique(y_train))

        # Initialize arrays to store class means, covariances, and priors
        class_means = []
        class_covariances = []
        class_priors = []

        # Loop through each class
        for c in range(num_classes):
            # Select data points belonging to class c
            class_c_data = X_train[y_train == c]

            # Flatten the 3D array to 2D
            class_c_data_2d = class_c_data.reshape(class_c_data.shape[0], -1)

            # Calculate class mean
            class_mean = np.mean(class_c_data_2d, axis=0)
            class_means.append(class_mean)

            # Calculate class covariance matrix
            class_covariance = np.cov(class_c_data_2d, rowvar=False)
            class_covariances.append(class_covariance)

            # Calculate class prior
            class_prior = len(class_c_data) / len(X_train)
            class_priors.append(class_prior)

        # Convert lists to numpy arrays
        self.class_means = np.array(class_means)
        self.class_covariances = np.array(class_covariances)
        self.class_priors = np.array(class_priors)

    def predict(self, X_test):
        # Initialize an array to store predicted labels
        predictions = []

        # Loop through each data point in the test set
        for sample in X_test:
            # Flatten the input image to a 1D array
            sample_1d = sample.flatten()

            # Calculate the quadratic discriminant function for each class
            qda_scores = []
            for i in range(len(self.class_means)):
                class_mean = self.class_means[i]
                class_covariance = self.class_covariances[i]

                # Mahalanobis distance
                mahalanobis_distance = np.dot(np.dot((sample_1d - class_mean).T, np.linalg.inv(class_covariance)), (sample_1d - class_mean))

                # Quadratic discriminant function with log determinant
                qda_score = -0.5 * mahalanobis_distance - 0.5 * np.linalg.slogdet(class_covariance)[1] + np.log(self.class_priors[i])
                qda_scores.append(qda_score)

            # Predict the class with the maximum QDA score
            predicted_class = np.argmax(qda_scores)
            predictions.append(predicted_class)

        # Convert list to numpy array
        return np.array(predictions)



from utils import load_and_prepare_data
from sklearn.metrics import accuracy_score

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
