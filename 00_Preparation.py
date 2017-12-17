#----------------------------------------------------
# Import libraries
import numpy as np
import pandas as pd

#----------------------------------------------------
# Import data using pandas
data = pd.read_csv("~/Dropbox/github/py/python_ML/airline_weather.csv", sep = ",")

# View data
data.values

#----------------------------------------------------
# Create dummy matrix for categorial predictors using pandas 

# Recode Origin into binary 
data["Origin"] = pd.Categorical.from_array(data["Origin"]).codes

# Create binary matrix for carrier
carrier_dummies = pd.get_dummies(data.ix[:, "UniqueCarrier"])
# Rename all columns
carrier_dummies.columns = ["Carrier_" + x for x in carrier_dummies.columns.values]

# Build design matrix from original data and dummy matrix 
data = pd.concat([data, carrier_dummies], axis = 1)

#----------------------------------------------------
# Split data in Train/Test Set
# Use train_test_split from SciKit Learn
from sklearn import cross_validation
data_train, data_test = cross_validation.train_test_split(data, test_size = 0.3)

#----------------------------------------------------
# Create Target and predictors

# Show column names
print(data_train.columns.values)

# Define a list of predictors
predictor_names = ["DepDelay", "TaxiOut", "TaxiIn", "AirTime", "Origin", "DepTime", "Distance"]
# Add dummies for carriers to list
predictor_names.extend(data_train.columns.values[43:26:-1]) 
print(predictor_names)
# Subset data by list
data_train_X = data_train[predictor_names]
data_test_X = data_test[predictor_names]

# Define target
data_train_Y = pd.Categorical.from_array(data_train["Delay"]).codes
data_test_Y = pd.Categorical.from_array(data_test["Delay"]).codes






