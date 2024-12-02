# A.I. Disclaimer: All work for this assignment was completed by myself
# and entirely without the use of artificial intelligence tools such as
# ChatGPT, MS Copilot, other LLMs, etc.


# Hide warnings
import warnings
warnings.filterwarnings("ignore")

# General use
import pandas as pd
import numpy as np
import time
import math
import os

# Model training and testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Models
from sklearn.ensemble import RandomForestClassifier  # For control comparison
from sklearn.tree import DecisionTreeClassifier

# MPI section
# from mpi4py import MPI

# Multiprocessing section
# import multiprocessing
# from concurrent.futures.process import ProcessPoolExecutor
# import concurrent
# from tempfile import mkdtemp
# import os.path as path


# #### Loading the dataset(s)
print('\nLoading data..\n')

main_data = pd.read_csv('Covid Dataset.csv')

print(main_data.head())
print(main_data.info())

# Example using Label Encoding (for binary categorical variables)
label_encoder = LabelEncoder()

# List of columns to encode
cols_to_encode = main_data.columns  # Since all columns are objects

# Apply label encoding to each column
for col in cols_to_encode:
    main_data[col] = label_encoder.fit_transform(main_data[col])

# Check the transformed data
print(main_data.head())

# Assuming 'COVID-19' is the target variable
X = main_data.drop('COVID-19', axis=1)
y = main_data['COVID-19']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('\nTraining models...\n')


# #### Using ML to train and make predictions 
# #### Control random forest
main_data.info()

# Initialize the Random Forest model
n_estimators = int(os.environ['MPI_COMM_WORLD_SIZE'])
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=5)

# Start timing for training
train_start_time = time.time()
model.fit(X_train, y_train)
train_end_time = time.time()

# Calculate and print training time
train_time = train_end_time - train_start_time
print(f"Training time: {train_time:.4f} seconds")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)