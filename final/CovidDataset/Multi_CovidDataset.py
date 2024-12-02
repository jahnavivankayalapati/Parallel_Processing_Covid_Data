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
# from mpi4py import MPI

# Multiprocessing section
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
import concurrent
from tempfile import mkdtemp
import os.path as path


# #### Loading the dataset(s)
print('\nLoading data...\n')

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

# Split data into training and testing chunks
train_data = main_data.iloc[0:math.floor(main_data.shape[0] * .8)]
test_data = main_data.iloc[math.ceil(main_data.shape[0] * .8):]

# Split testing data
X_test = test_data.drop('COVID-19', axis=1)
y_test = test_data['COVID-19']

# Free memory
del main_data, test_data


print('\nTraining Models...\n')


# #### Using ML to train and make predictions 
# #### Multi-Processing

# Function to train a Decision Tree model
def train_model(name, start, stop):
    # Setup memmap data as np array
    data = np.memmap(name, dtype=ARRAY_TYPE, shape=ARRAY_SHAPE)

    # Recreate a dataframe from the data
    df = pd.DataFrame(data=data[start:stop], columns=COLUMNS)
    # print(df)
    
    # Split the data into features and target variable
    X = df.drop(columns=['COVID-19'])
    y = df['COVID-19']
    
    # Fit a decision tree on the data
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X, y)
    
    return clf

# Create a numpy array of the prepared dataset
data_array = train_data.to_numpy()

# Number of processes to run
NUM_WORKERS = int(os.environ['MPI_COMM_WORLD_SIZE'])
# NUM_WORKERS = 32
# print(NUM_WORKERS)
    
# Set up values for training
futures = []
COLUMNS = train_data.columns
ARRAY_SIZE = data_array.shape[0]
ARRAY_SHAPE = data_array.shape
ARRAY_TYPE = data_array.dtype
chunk_size = int(ARRAY_SIZE / NUM_WORKERS)
# print(data_array)
# print(ARRAY_SIZE, NUM_WORKERS, chunk_size)

# Create a memmap as a temporary file in memory and copy the main_data to it
filename = path.join(mkdtemp(), 'array.dat')
dst = np.memmap(filename, dtype=ARRAY_TYPE, mode='w+', shape=ARRAY_SHAPE)
np.copyto(dst, data_array)
# print(dst)

train_start_time = time.time()

# Start training for all processes
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for i in range(0, NUM_WORKERS):
        # Maps entire range of data array to processes in chunk size
        if i == 0:
            start = 0
            end = chunk_size
        elif i == NUM_WORKERS - 1:
            start = ((i - 1) * chunk_size + chunk_size)
            end = ARRAY_SIZE
        else:
            start = ((i - 1) * chunk_size + chunk_size) + 1
            end = start + chunk_size - 1
        # print(i, start, end)
        futures.append(executor.submit(train_model, filename, start, end))
futures, _ = concurrent.futures.wait(futures)

train_end_time = time.time()

print('Training time:', train_end_time - train_start_time)

# Collect trained models
models = []
for i, future in enumerate(futures):
    # print(future.result())
    # y_pred = future.result().predict(main_data.drop(columns='COVID-19'))
    # accuracy = accuracy_score(main_data['COVID-19'], y_pred)
    # print(f'Overall Accuracy for model {i}:', accuracy)
    models.append(future.result())

# Create an ensemble of models (optional)
y_pred = np.zeros(y_test.shape)
for m in models:
    y_pred += m.predict(X_test)
y_pred = np.where(y_pred > (NUM_WORKERS / 2), 1, 0)  # Majority voting

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Overall Accuracy:", accuracy)

# Calculate testing time
test_start_time = time.time()
final_user_input = np.array([X_test.iloc[0]])  # Example input; replace as needed
user_pred = np.zeros(final_user_input.shape[0])
for m in models:
    user_pred += m.predict(final_user_input)
user_pred = np.where(user_pred > (NUM_WORKERS / 2), 1, 0)  # Majority voting
test_end_time = time.time()

# Calculate and print testing time
test_time = test_end_time - test_start_time
print(f"Testing time: {test_time:.4f} seconds")

# Output results as Yes or No
prediction = 'Yes' if user_pred[0] == 1 else 'No'
print("Predicted Disease Present:", prediction)