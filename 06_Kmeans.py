#----------------------------------------------------
# Import libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#----------------------------------------------------
# Import data using pandas
data = pd.read_csv("~/Desktop/Playground/python_ML/airline_weather.csv", sep = ",")

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

# Define list of cluster variables
cluster_names = ["Temp", "Humidity", "Pressure"]

# Subset data by variable list
data_cluster = data[cluster_names]

#----------------------------------------------------
# Kmeans with n = 5 Clusters

# Number of centers
n = 5

# Create kmeans object and fitted model
kmeans_model = KMeans(n_clusters = n)
kmeans_fit = kmeans_model.fit(data_cluster)

# Get predicted Centers
cluster_labels = kmeans_model.fit_predict(data_cluster)

# Frequency table of cluster centers using scipy's itemfreq
import scipy
scipy.stats.itemfreq(cluster_labels)

#----------------------------------------------------
# Range of cluster centers using silhouettes
from sklearn.metrics import silhouette_samples, silhouette_score

# Range of cluster centers
range_n_clusters = [2, 3, 4, 5, 6]

# Loop for range of centers
for n_clusters in range_n_clusters:
    
    # Fit model
    kmeans_model = KMeans(n_clusters = n_clusters, random_state = 10)
    cluster_labels = kmeans_model.fit_predict(data_cluster)
    
    # Print silhouette
    silhouette_avg = silhouette_score(data_cluster, cluster_labels)
    print("For n = ", n_clusters, "cluster, the average silhouette score is :", silhouette_avg)
