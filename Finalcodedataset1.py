# ******************************************************************************
# * FILE: Finalcodedataset1.py
# * 
# * DESCRIPTION:
# * This program performs parallel training and testing of a Random Forest model
# * using MPI (Message Passing Interface) and various parallelization strategies.
# * It compares the performance of Multi-Processing, SPMD, MPI, and MPMD methods.
# *
# * AUTHOR: Kuncham Padma Priyanka
# * LAST REVISED: 12/01/2024
# ******************************************************************************

import warnings
warnings.filterwarnings("ignore")  # Suppress warning messages
import pandas as pd
import numpy as np
import time
#  MPI for parallel processing
from mpi4py import MPI  
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier  
# For splitting data and hyperparameter tuning
from sklearn.model_selection import train_test_split, RandomizedSearchCV 
# For parallel processing with multi-processing
from multiprocessing import Pool  

# Load the dataset
main_data = pd.read_csv('Covid Dataset.csv')  

# Encode categorical labels into numerical format
from sklearn.preprocessing import LabelEncoder, StandardScaler
label_encoder = LabelEncoder()
# Process each column
for col in main_data.columns:  
    main_data[col] = label_encoder.fit_transform(main_data[col])

# Standardize features to have a mean of 0 and variance of 1
scaler = StandardScaler()
X = main_data.drop('COVID-19', axis=1)  
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
y = main_data['COVID-19']  

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MPI communication
comm = MPI.COMM_WORLD  
rank = comm.Get_rank()  
size = comm.Get_size()  

# Function to create an optimized Random Forest using RandomizedSearchCV
def optimized_random_forest(X, y):
    # Define the model and hyperparameters to test
    # Enable parallelism within Random Forest
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)  
    param_grid = {
        'n_estimators': [50, 100], 
        'max_depth': [10, None],  
        'min_samples_split': [5, 10], 
    }
    # Perform RandomizedSearchCV for faster tuning
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_grid,
        # Limit the number of random samples
        n_iter=10, 
        cv=3,
        scoring='accuracy',
        random_state=42,
        # Parallelize hyperparameter search
        n_jobs=-1  
    )
    # Train the model on the data
    random_search.fit(X, y)  
    # Return the best model
    return random_search.best_estimator_  

# Function to evaluate model performance and measure testing time
def evaluate_model(clf, X_test, y_test):
    start_time = time.time()  
    y_pred = clf.predict(X_test) 
    end_time = time.time()  
    test_time = end_time - start_time  
    print(f"Testing Time: {test_time:.4f} seconds")
    return test_time  

# Multi-Processing: Use Pool for parallel training
def train_with_multiprocessing():
    print("\nTraining with Multi-Processing:")
    start_time = time.time()  
    # Split training data across processes and train in parallel
    pool = Pool(processes=size)  
    models = pool.starmap(optimized_random_forest, zip(np.array_split(X_train, size), np.array_split(y_train, size)))
    pool.close()  
    pool.join()  
    # Calculate training time
    train_end_time = time.time() - start_time  
    print(f"Training Time with Multi-Processing: {train_end_time:.4f} seconds")
    # Evaluate the model performance
    test_time = evaluate_model(models[0], X_test, y_test)
    print(f"Total Time (Training + Testing) with Multi-Processing: {train_end_time + test_time:.4f} seconds")
    return train_end_time, test_time

# SPMD: Single Program Multiple Data
def train_with_spmd():
    print("\nTraining with SPMD:")
    X_train_split = np.array_split(X_train, size)[rank] 
    y_train_split = np.array_split(y_train, size)[rank] 
    start_time = time.time()  
    clf = optimized_random_forest(X_train_split, y_train_split)  
    train_end_time = time.time() - start_time 
    print(f"Rank {rank}: Training Time with SPMD: {train_end_time:.4f} seconds")
    # Gather models at rank 0
    models = comm.gather(clf, root=0)
    if rank == 0:
        test_time = evaluate_model(models[0], X_test, y_test) 
        print(f"Total Time (Training + Testing) with SPMD: {train_end_time + test_time:.4f} seconds")
        return train_end_time, test_time
    return None, None

# MPI: Fully distributed parallelism
def train_with_mpi():
    print("\nTraining with MPI:")
    X_train_split = np.array_split(X_train, size)[rank]  
    y_train_split = np.array_split(y_train, size)[rank]
    start_time = time.time() 
    clf = optimized_random_forest(X_train_split, y_train_split) 
    train_end_time = time.time() - start_time  
    print(f"Rank {rank}: Training Time with MPI: {train_end_time:.4f} seconds")
    # Gather results at root rank
    models = comm.gather(clf, root=0)
    if rank == 0:
        test_time = evaluate_model(models[0], X_test, y_test) 
        print(f"Total Time (Training + Testing) with MPI: {train_end_time + test_time:.4f} seconds")
        return train_end_time, test_time
    return None, None

# MPMD: Multiple Programs Multiple Data
def train_with_mpmd():
    print("\nTraining with MPMD:")
    start_time = time.time()  
    if rank == 0:
        print("Rank 0 is managing training.")
        clf = optimized_random_forest(X_train, y_train)  
    comm.Barrier()  # Wait for all ranks to finish
    train_end_time = time.time() - start_time if rank == 0 else 0.0  
    if rank == 0:
        test_time = evaluate_model(clf, X_test, y_test) 
        print(f"Training Time with MPMD: {train_end_time:.4f} seconds")
        print(f"Total Time (Training + Testing) with MPMD: {train_end_time + test_time:.4f} seconds")
        return train_end_time, test_time
    return None, None

# Run models and display results
if rank == 0:
    multiprocessing_train_time, multiprocessing_test_time = train_with_multiprocessing()
    spmd_train_time, spmd_test_time = train_with_spmd()
    mpi_train_time, mpi_test_time = train_with_mpi()
    mpmd_train_time, mpmd_test_time = train_with_mpmd()
    print("\nSummary of Time Complexity:")
    print(f"Multi-Processing Training Time: {multiprocessing_train_time:.4f} seconds | Testing Time: {multiprocessing_test_time:.4f} seconds")
    print(f"SPMD Training Time: {spmd_train_time:.4f} seconds | Testing Time: {spmd_test_time:.4f} seconds")
    print(f"MPI Training Time: {mpi_train_time:.4f} seconds | Testing Time: {mpi_test_time:.4f} seconds")
    print(f"MPMD Training Time: {mpmd_train_time:.4f} seconds | Testing Time: {mpmd_test_time:.4f} seconds")

MPI.Finalize()
