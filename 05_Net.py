#----------------------------------------------------
# Import MLP from Scikit Learn (not included in v0.17...)
from sklearn.neural_network import MLPClassifier

# Create MLP Object
mlp_model = MLPClassifier(algorithm = "l-bfgs", alpha = 1e-5, hidden_layer_sizes = (5, 2), random_state = 1)

# Fit SVC using test data
mlp_fit = MLPClassifier.fit(data_train_X, data_train_Y)

# Accuracy on training data
print(svc_fit.score(data_train_X, data_train_Y))


