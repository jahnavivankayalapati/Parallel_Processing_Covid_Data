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


# #### Loading the dataset(s)
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print('\nLoading data...\n')

# Check shape for creating buffers
main_shape = (1000000, 117)
train_shape = (math.ceil(main_shape[0]*.8), main_shape[1]-1)
X_chunk_shape = (math.ceil(train_shape[0]/size), train_shape[1])
y_chunk_shape = (X_chunk_shape[0],)
y_cols = ['new_confirmed']

# Define variables for all workers
X_cols = None
X_train = np.zeros(X_chunk_shape)
y_train = np.zeros(y_chunk_shape)
X_train_splits = None
y_train_splits = None

# Have the main worker load and preprocess the data
if rank == 0:    
    main_data = pd.read_csv('1mrows.csv')
    main_data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
    
    # Fill in missing data, assumed no new cases, for target
    main_data['new_confirmed'] = main_data['new_confirmed'].fillna(0)
    
    # Example using Label Encoding (for binary categorical variables)
    label_encoder = LabelEncoder()
    
    # List of object columns to encode
    cols_to_encode = main_data.select_dtypes('object')
    
    # Apply label encoding to each column
    for col in cols_to_encode:
        main_data[col] = label_encoder.fit_transform(main_data[col])
    
    # Check the transformed data
    print(main_data.info(verbose=True, show_counts=True))
    
    # Assuming 'new_confirmed' is the target variable
    X = main_data.drop('new_confirmed', axis=1)
    y = main_data['new_confirmed']
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_cols = X_train.columns

    # Create subsets of the data
    X_train_splits = np.array_split(X_train, size)
    y_train_splits = np.array_split(y_train, size)

    # Assign root's task
    X_train = X_train_splits[0]
    y_train = y_train_splits[0]
    print(X_train.info())

    # Allow memory to be free'd
    del main_data, X, y


if rank == 0:
    print('\nTraining models...\n')


# Send the data to all workers
# Used for reconstructing DFs
X_cols = comm.bcast(X_cols, root=0)
# Pass X data
if rank == 0:
    for i in range(1, size):
        comm.Send(X_train_splits[i].to_numpy(), dest=i)
else:
    comm.Recv(X_train, source=0)
    # Reconstruct DF
    X_train = pd.DataFrame(data=X_train, columns=X_cols)
# Pass y data
if rank == 0:
    for i in range(1, size):
        comm.Send(y_train_splits[i].to_numpy(), dest=i)
else:
    comm.Recv(y_train, source=0)
    # Reconstruct DF
    y_train = pd.DataFrame(data=y_train, columns=y_cols)

# Initialize the Random Forest model
model = DecisionTreeClassifier(max_depth=5)

# Start timing for training
train_start_time = time.time()
model.fit(X_train, y_train)

# Gather the trained models to the root process finishing training
models = comm.gather(model, root=0)
train_end_time = time.time()
train_time = train_end_time - train_start_time

# Evaluate the model on the test set (only on rank 0)
if rank == 0:
    # Output training time
    print(f"Process {rank}: Training time: {train_time:.4f} seconds")
    
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
    print(f"Testing time: {test_time:.4f} seconds")

    # Output results as Yes or No
    prediction = 'Yes' if user_pred[0] == 1 else 'No'
    print("Predicted Disease Present:", prediction)

# Finalize MPI
MPI.Finalize()