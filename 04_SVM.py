#----------------------------------------------------
# Import SVM from Scikit Learn
from sklearn import svm
from sklearn import preprocessing

# scikit-learn.org recomments feature scaling 
data_train_X_scaled = preprocessing.scale(data_train_X)
data_test_X_scaled = preprocessing.scale(data_test_X)

# Create Support Vector Machine with radial kernel (default)
svc_model = svm.SVC(kernel = "rbf")

# Fit SVC using test data
svc_fit = svc_model.fit(data_train_X_scaled, data_train_Y)

# Accuracy on training data
print(svc_fit.score(data_train_X_scaled, data_train_Y))

# Get support vectors
svc_model.support_vectors_

# Get indices of support vectors
svc_model.support_

# Get predicted probabilities per class (rebuild model object with appropriate option, fit model, and get probs)
svc_model = svm.SVC(probability = True)
svc_fit = svc_model.fit(data_train_X_scaled, data_train_Y)
svc_fit.predict_proba(data_train_X_scaled)

#----------------------------------------------------
# Use model object to predict test data

# Predicted classes and probs
svc_model = svm.SVC()
svc_fit = svc_model.fit(data_train_X_scaled, data_train_Y)
svc_predicted = svc_fit.predict(data_test_X_scaled)

# Import fcts for evaluation from sklean
from sklearn import metrics

# Generate evaluation metrics
print(metrics.accuracy_score(data_test_Y, svc_predicted))

#----------------------------------------------------
# Compare different kernels

C = .5

# Linear Support Vector Classifier with LinearSVC (alternative SVC with linear kernel)
lin_1_svc = svm.LinearSVC().fit(data_train_X_scaled, data_train_Y)
lin_2_svc = svm.SVC(kernel = 'linear', C = C, cache_size = 1000).fit(data_train_X_scaled, data_train_Y) 
# Polynomial Kernel
pol_svc = svm.SVC(kernel = 'poly', C = C, cache_size = 1000, degree = 2).fit(data_train_X_scaled, data_train_Y) 
# Radient base function
rbf_svc = svm.SVC(kernel = 'rbf', C = C, cache_size = 1000, gamma = .5).fit(data_train_X_scaled, data_train_Y) 
# Sigmoid
sig_svc = svm.SVC(kernel = 'sigmoid', C = C, cache_size = 1000, coef0 = 1).fit(data_train_X_scaled, data_train_Y) 


# Predictions and Accuracies
print(metrics.accuracy_score(data_test_Y, lin_1_svc.predict(data_test_X_scaled)))
print(metrics.accuracy_score(data_test_Y, lin_2_svc.predict(data_test_X_scaled)))
print(metrics.accuracy_score(data_test_Y, pol_svc.predict(data_test_X_scaled)))
print(metrics.accuracy_score(data_test_Y, rbf_svc.predict(data_test_X_scaled)))
print(metrics.accuracy_score(data_test_Y, sig_svc.predict(data_test_X_scaled)))



