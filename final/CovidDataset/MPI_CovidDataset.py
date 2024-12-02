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
# from sklearn.ensemble import RandomForestClassifier  # For control comparison
from sklearn.tree import DecisionTreeClassifier

# MPI section
from mpi4py import MPI

# Multiprocessing section
# import multiprocessing
# from concurrent.futures.process import ProcessPoolExecutor
# import concurrent
# from tempfile import mkdtemp
# import os.path as path


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# #### Loading the dataset(s)
if rank == 0:
    print('\nLoading data...\n')

main_data = pd.read_csv('Covid Dataset.csv')

# Example using Label Encoding (for binary categorical variables)
label_encoder = LabelEncoder()

# List of columns to encode
cols_to_encode = main_data.columns  # Since all columns are objects

# Apply label encoding to each column
for col in cols_to_encode:
    main_data[col] = label_encoder.fit_transform(main_data[col])

# Check the transformed data
# print(main_data.head())

# Assuming 'COVID-19' is the target variable
X = main_data.drop('COVID-19', axis=1)
y = main_data['COVID-19']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# #### Using ML to train and make predictions 
# ####  MPI
if rank == 0:
    print('\nTraining models...\n')

# Split the training data among processes
X_train_split = np.array_split(X_train, size)[rank]
y_train_split = np.array_split(y_train, size)[rank]

# Initialize the Random Forest model
model = DecisionTreeClassifier(max_depth=5)

# Start timing for training
train_start_time = time.time()
model.fit(X_train_split, y_train_split)

# Gather the trained models to the root process
models = comm.gather(model, root=0)
train_end_time = time.time()
train_time = train_end_time - train_start_time

# Evaluate the model on the test set (only on rank 0)
if rank == 0:
    # Create an ensemble of models (optional)
    y_pred = np.zeros(y_test.shape)
    for m in models:
        y_pred += m.predict(X_test)
    y_pred = np.where(y_pred > (size / 2), 1, 0)  # Majority voting

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Overall Accuracy:", accuracy)

    # Calculate testing time
    test_start_time = time.time()
    final_user_input = np.array([X_test.iloc[0]])  # Example input; replace as needed
    user_pred = np.zeros(final_user_input.shape[0])
    for m in models:
        user_pred += m.predict(final_user_input)
    user_pred = np.where(user_pred > (size / 2), 1, 0)  # Majority voting
    test_end_time = time.time()

    # Calculate and print testing time
    test_time = test_end_time - test_start_time
    print(f"Training time: {train_time:.4f} seconds")
    print(f"Testing time: {test_time:.4f} seconds")

    # Output results as Yes or No
    prediction = 'Yes' if user_pred[0] == 1 else 'No'
    print("Predicted Disease Present:", prediction)

# Finalize MPI
MPI.Finalize()