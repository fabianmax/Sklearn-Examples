
#----------------------------------------------------
# Run Logistic regression from Scikit Learn
from sklearn import linear_model

# Create logit regression object
glm_model = linear_model.LogisticRegression()

# Fit logit using test data
glm_fit = glm_model.fit(data_train_X, data_train_Y)

# Accuracy on training data
print(glm_fit.score(data_train_X, data_train_Y))

# Show coefficients
print(pd.DataFrame(data_train_X.columns, np.transpose(glm_fit.coef_)))

#----------------------------------------------------
# Use model object to predict test data

# Predicted classes and probs
glm_predicted = glm_fit.predict(data_test_X)
glm_probs = glm_fit.predict_proba(data_test_X)

# Import fcts for evaluation from sklean
from sklearn import metrics

# Generate evaluation metrics
print(metrics.accuracy_score(data_test_Y, glm_predicted))
print(metrics.confusion_matrix(data_test_Y, glm_predicted))





