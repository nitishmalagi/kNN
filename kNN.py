# Data Programming Project
# k nearest neighbours on the TunedIT data set

# Import packages
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import numpy.random as npr


# Loading in the data using the pandas read_csv function. The last variable 
# 'RockOrNot' determines whether the music genre for that sample is rock or not
tunedit_df = pd.read_csv('tunedit_genres.csv')
percentage_of_rock_songs = tunedit_df['RockOrNot'].value_counts(normalize=True)
print(percentage_of_rock_songs[1] * 100)

# 48.83% of the songs in the dataset are rock songs


# To perform a classification algorithm, we need to define a classification 
# variable and separate it from the other variables. We will use 'RockOrNot' as 
# our classification variable.Below is the code to separate the data into a 
# DataFrames X and a Series y, where X contains a standardised version of 
# everything except for the classification variable ('RockOrNot'), and y contains 
# only the classification variable. To standardise the variables in X, we 
# subtract the mean and divide by the standard deviation
temp = tunedit_df.drop('RockOrNot',axis=1)
y = tunedit_df['RockOrNot']
X = (temp - temp.mean())/(temp.std())


# Finding correlation of variables in X with y
correlation = X.corrwith(y, axis=0)
print(correlation.idxmax(axis=1))
# The variable PAR_SFM_M has the largest correlation(0.596) with y


# When performing a classification problem, we need to fit the model to a portion of 
# data, and use the remaining data to determine how good the model fit was.
# Below is the code to divide X and y into training and test sets, using 75%
# of the data for training and keep 25% for testing. The data is randomly
# selected. Additionally, the data in the training set does not appear in the test set, 
# and vice versa, so that when recombined, all data is accounted for. 
# The seed value 123 is used while generating random numbers.
X_train = X.sample(frac=0.75, random_state=123)
y_train = y.sample(frac=0.75, random_state=123)
X_test = X.drop(X_train.index)
y_test = y.drop(y_train.index)

train_set = pd.concat([X_train,y_train], axis=1, sort=False)
test_set = pd.concat([X_test,y_test], axis=1, sort=False)


# Checking the percentage of rock songs in the training dataset and in the test dataset.
train_rock_songs = ((train_set['RockOrNot'].sum()/ train_set['RockOrNot'].count()) * 100)
print(train_rock_songs)
test_rock_songs = ((test_set['RockOrNot'].sum()/ test_set['RockOrNot'].count()) * 100)
print(test_rock_songs)
# The percentage of Rock songs in train dataset is 49.41% which is greater than seen before
# The percentage of Rock songs in test dataset is 47.09%  which is lesser than seen before




# Below is a function to run kNN on the data sets. kNN works by the following algorithm:
# 1) Choose a value of k (usually odd)
# 2) For each observation, find its k closest neighbours
# 3) Take the majority vote (mean) of these neighbours
# 4) Classify observation based on majority vote

# We're going to use standard Euclidean distance to find the distance between 
# observations, defined as sqrt( (xi - xj)^T (xi-xj) )
# A useful short cut for this is the scipy functions pdist and squareform

# The function inputs are:
# - DataFrame X of explanatory variables 
# - binary Series y of classification values 
# - value of k (you can assume this is always an odd number)

# The function should produces Series y_star of predicted classification values

from scipy.spatial.distance import pdist, squareform
from statistics import mean

def kNN(X,y,k):
    # Find the number of obsvervation
    n = X.shape[0]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Set up return values
    y_star = []
    # Calculate the distance matrix for the observations in X
    dist =  squareform(pdist(X, 'euclidean'))
    # Make all the diagonals very large so it can't choose itself as a
    # closest neighbour
    np.fill_diagonal(dist, 99999)
    # Loop through each observation to create predictions
    for i in range(n):
        neighbours = []
        for j in dist[i].argsort()[:k]:
            neighbours.append(y[j])
        # Find the y values of the k nearest neighbours
        if sum(neighbours) > (k/2):
            y_star.append(1)
        else:
            y_star.append(0)
         
    return pd.Series(y_star)

y12 = kNN(X_train, y_train, 3)



# The misclassification rate is the percentage of times the output of a 
# classifier doesn't match the classification value. 
# Below is the misclassification rate of the kNN classifier for X_train and y_train, with k=3
def misclassification_rate(y, predictions):
    correct_predictions = 0
    y = y.reset_index(drop=True)
    for i, val in enumerate(y):
        if predictions[i] == y[i]:
            correct_predictions += 1
    return ((1 - (correct_predictions / len(y))) * 100)

predictions = kNN(X_train, y_train, 3)
print(misclassification_rate(y_train, predictions))

# Misclassification Rate is 4.7%


        
# The best choice for k depends on the data. Below is a function kNN_select that 
# will run a kNN classification for a range of k values, and compute the 
# misclassification rate for each.

# The function inputs are:
# - DataFrame X of explanatory variables 
# - binary Series y of classification values 
# - a list of k values k_vals

# The function produces a Series mis_class_rates, indexed by k, with the misclassification rates for 
# each k value in k_vals.
def kNN_select(X,y,k_vals):
    rates = []
    for k in k_vals:
        predictions = kNN(X, y, k)
        rates.append(misclassification_rate(y, predictions)) 
    misclassification_rates = pd.Series(rates, index = k_vals) 
    return misclassification_rates



# Running the function kNN_select on the training data for k = [1, 3, 5, 7, 9] 
# to find the value of k with the best misclassification rate. Then, the best 
# value of k is used to report the mis-classification rate for the test data.
result = kNN_select(X_train, y_train, k_vals=[1, 3, 5, 7, 9])
print('Best value of k is ', result.idxmin())
print('with misclassification rate of ',result.min())
# The best value of k is 1 with misclassification rate of 3.33% 



# Below is a function to generalise the k nearest neighbours classification 
# algorithm. The function :
# - Separates out the classification variable for the other variables in the dataset
# - Divides X and y into training and test set, where the number in each is 
#   specified by 'percent_train'.
# - Runs the k nearest neighbours classification on the training data, for a set 
#   of k values, computing the mis-classification rate for each k
# - Finds the k that gives the lowest mis-classification rate for the training data,
#   and hence, the classification with the best fit to the data.
# - Uses the best k value to run the k nearest neighbours classification on the test
#   data, and calculates the mis-classification rate
# The function returns the mis-classification rate for a k nearest neighbours
# classification on the test data, using the best k value for the training data
def kNN_classification(df,class_column,seed,percent_train,k_vals):
    # df            - DataFrame to 
    # class_column  - column of df to be used as classification variable, should
    #                 specified as a string  
    # seed          - seed value for creating the training/test sets
    # percent_train - percentage of data to be used as training data
    # k_vals        - set of k values to be tests for best classification
    
    # Separate X and y
    temp = df.drop(class_column, axis=1)
    y = df[class_column]
    X = (temp - temp.mean()) / (temp.std())

    # Divide into training and test
    X_train = X.sample(frac = percent_train, random_state = seed)
    y_train = y.sample(frac = percent_train, random_state = seed)
    X_test = X.drop(X_train.index)
    y_test = y.drop(y_train.index)

    # Compute the mis-classification rates for each for the values in k_vals
    misclassification_rates = kNN_select(X_train, y_train, k_vals)

    # Find the best k value, by finding the minimum entry of mis_class_rates 
    best_k = misclassification_rates.idxmin()

    # Run the classification on the test set to see how well the 'best fit'
    # classifier does on new data generated from the same source
    predictions_test_set = kNN(X_test, y_test, best_k)

    # Calculate the mis-classification rates for the test data
    misclassification_rates_test = misclassification_rate(y_test, predictions_test_set)

    return misclassification_rates_test
      
# Testing the function with the TunedIT data set
print(kNN_classification(tunedit_df, 'RockOrNot', 123, 0.75, [1, 3, 5, 7, 9]))




# Now testing the function with another dataset, to ensure that the code has
# generalised using the house_votes.csv dataset, with 'Party' as the 
# classifier. 
# This dataset contains the voting records of 435 congressman and women in the 
# US House of Representatives. The parties are specified as 1 for democrat and 0
# for republican, and the votes are labelled as 1 for yes, -1 for no and 0 for
# abstained.
house_df = pd.read_csv('house_votes.csv')
print(kNN_classification(house_df, 'Party', 123, 0.75, [1, 3, 5, 7, 9]))
# The misclassification rate is 8.25 with k = 1 
