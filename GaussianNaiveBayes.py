import numpy as np
from utils import load_and_prepare_data
from sklearn.metrics import accuracy_score

class GaussianNaiveBayes:
    def __init__(self):
        # Initialize any required variables
        self.class_means = None
        self.class_variances = None
        self.class_priors = None

    def fit(self, X_train, y_train):
        # Number of classes
        num_classes = len(np.unique(y_train))

        # Initialize arrays to store class means, variances, and priors
        class_means = []
        class_variances = []
        class_priors = []

        # Loop through each class
        for c in range(num_classes):
            # Select data points belonging to class c
            class_c_data = X_train[y_train == c]

            # Calculate class mean and variance
            class_mean = np.mean(class_c_data, axis=0)
            class_variance = np.var(class_c_data, axis=0)

            # Store class mean and variance
            class_means.append(class_mean)
            class_variances.append(class_variance)

            # Calculate class prior
            class_prior = len(class_c_data) / len(X_train)
            class_priors.append(class_prior)

        # Convert lists to numpy arrays
        self.class_means = np.array(class_means)
        self.class_variances = np.array(class_variances)
        self.class_priors = np.array(class_priors)

    def predict(self, X_test):
        # Initialize an array to store predicted labels
        predictions = []

        # Loop through each data point in the test set
        for sample in X_test:
            # Calculate the log likelihood for each class
            log_likelihoods = []
            for i in range(len(self.class_means)):
                class_mean = self.class_means[i]
                class_variance = self.class_variances[i]

                # Gaussian log likelihood
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * class_variance) + ((sample - class_mean) ** 2) / class_variance)
                log_likelihoods.append(log_likelihood)

            # Calculate the log posterior probability for each class
            log_posterior = np.log(self.class_priors) + np.array(log_likelihoods)

            # Predict the class with the maximum log posterior probability
            predicted_class = np.argmax(log_posterior)
            predictions.append(predicted_class)

        # Convert list to numpy array
        return np.array(predictions)





print("\n****************************************")
print("*     Gaussian Naive Bayes Solution    *")
print("****************************************")

print("Loading data...",end="")
X_train, y_train, X_test, y_test = load_and_prepare_data()
print("done.")

# Instantiate Gaussian Naive Bayes model
gnb_model = GaussianNaiveBayes()


# Fit the model to the training data
print("fitting Gaussian Naive Bayes model...",end="")
gnb_model.fit(X_train, y_train)
print("done.")

# Make predictions on the test data
predictions = gnb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy of Gaussina Naive Bayes model : {accuracy}')
