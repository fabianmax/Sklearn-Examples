#----------------------------------------------------
# Libraries
from sklearn import tree

#----------------------------------------------------
# Run Tree classifier from Scikit Learn

# Create DecisionTree object and train
tree_model = tree.DecisionTreeClassifier()
tree_fit = tree_model.fit(data_train_X, data_train_Y)

# Accuracy on training data
print(tree_fit.score(data_train_X, data_train_Y))


#----------------------------------------------------
# Use model object to predict test data

# Predicted classes and probs
tree_predicted = glm_fit.predict(data_test_X)
tree_probs = glm_fit.predict_proba(data_test_X)

# Import fcts for evaluation from sklean
from sklearn import metrics

# Generate evaluation metrics
print(metrics.accuracy_score(data_test_Y, tree_predicted))
print(metrics.confusion_matrix(data_test_Y, tree_predicted))



#----------------------------------------------------
# Visualize tree

# ...
