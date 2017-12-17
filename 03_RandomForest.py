#----------------------------------------------------
# Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

#----------------------------------------------------
# Run Random Forest from Scikit Learn

# Create Random Forest object and train

forest_model = RandomForestClassifier(n_estimators = 50,
                                      max_depth = None,
                                      max_features = "sqrt",
                                      n_jobs = 2)    
forest_fit = forest_model.fit(data_train_X, data_train_Y)

# Cross-validated error on train data
scores = cross_val_score(forest_fit, data_train_X, data_train_Y)
print(scores.mean())

# Feature importance 
print(forest_fit.feature_importances_) 

#----------------------------------------------------
# Use model object to predict test data

# Predicted classes and probs
forest_predicted = forest_fit.predict(data_test_X)
forest_probs = forest_fit.predict_proba(data_test_X)

# Import fcts for evaluation from sklean
from sklearn import metrics

# Generate evaluation metrics
print(metrics.accuracy_score(data_test_Y, forest_predicted))
print(metrics.confusion_matrix(data_test_Y, forest_predicted))
