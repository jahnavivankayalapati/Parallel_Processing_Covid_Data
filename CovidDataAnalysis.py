



import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import time

# Model training and testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Models
from sklearn.ensemble import RandomForestClassifier  # For control comparison
from sklearn.tree import DecisionTreeClassifier

# MPI section
from mpi4py import MPI

# Multiprocessing section
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
import concurrent
import time
from tempfile import mkdtemp
import os.path as path


main_data = pd.read_csv('Covid Dataset.csv')

print(main_data.head())

print(main_data.info())

# Creating space in console output before next section
print('\n')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating space in console output before next section
print('\n')


# #### Using ML to train and make predictions 

# Initialize the Random Forest model
rf_model = RandomForestClassifier()

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Print the results
print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"Random Forest Precision: {precision_rf}")
print(f"Random Forest Recall: {recall_rf}")
print(f"Random Forest F1 Score: {f1_rf}")

# Creating space in console output before next section
print('\n')


# #### Control random forest
# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

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

# Testing the model with user input
user_input = ['continuous_sneezing', 'watering_from_eyes']  # Example symptoms
# Convert user input to a DataFrame (assuming it's consistent with training data)
user_input_df = pd.DataFrame({
    'Breathing Problem': [0],  # Example input (1: Yes, 0: No)
    'Fever': [0],
    'Dry Cough': [1],
    'Sore throat': [0],
    'Running Nose': [0],
    'Asthma': [0],
    'Chronic Lung Disease': [0],
    'Headache': [0],
    'Heart Disease': [0],
    'Diabetes': [0],
    'Hyper Tension': [0],
    'Fatigue': [1],
    'Gastrointestinal': [0],
    'Abroad travel': [0],
    'Contact with COVID Patient': [1],
    'Attended Large Gathering': [0],
    'Visited Public Exposed Places': [0],
    'Family working in Public Exposed Places': [0],
    'Wearing Masks': [1],
    'Sanitization from Market': [1]
})

# Ensure that the input columns match the training columns
final_user_input = user_input_df.reindex(columns=X_train.columns, fill_value=0)

# Start timing for testing
test_start_time = time.time()
user_pred = model.predict(final_user_input)
test_end_time = time.time()

# Calculate and print testing time
test_time = test_end_time - test_start_time
print(f"Testing time: {test_time:.4f} seconds")

# Decode the predicted class
predicted_class_label = label_encoder.inverse_transform(user_pred)[0]

# Convert predicted label to Yes/No
prediction_result = "Yes" if predicted_class_label == 1 else "No"
print("Predicted Disease (COVID-19 Positive):", prediction_result)

# Creating space in console output before next section
print('\n')

# #### Multiprocessing
# Function to train a Decision Tree model
def train_model(name, start, stop):
    # Setup memmap data as np array
    data = np.memmap(name, dtype=ARRAY_TYPE, shape=ARRAY_SHAPE)

    # Recreate a dataframe from the data
    df = pd.DataFrame(data=data[start:stop], columns=COLUMNS)
    # print('hello')
    # print(df)
    
    # Split the data into features and target variable
    X = df.drop(columns=['COVID-19'])
    y = df['COVID-19']
    
    # Fit a decision tree on the data
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    
    return clf

# Create a numpy array of the prepared dataset
data_array = main_data.to_numpy()

# Number of processes to run
NUM_WORKERS = multiprocessing.cpu_count()
    
# Set up values for training
futures = []
COLUMNS = main_data.columns
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
# Creating space in console output before next section
print('\n')
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Broadcast the data to all processes
main_data = comm.bcast(main_data, root=0)

# Split the data into features and target variable
X = main_data.drop(columns=["COVID-19"])
y = main_data["COVID-19"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Split the training data among processes
X_train_split = np.array_split(X_train, size)[rank]
y_train_split = np.array_split(y_train, size)[rank]

# Initialize the Random Forest model
model = DecisionTreeClassifier()

# Start timing for training
train_start_time = time.time()
model.fit(X_train_split, y_train_split)
train_end_time = time.time()

# Calculate and print training time
train_time = train_end_time - train_start_time
print(f"Process {rank}: Training time: {train_time:.4f} seconds")

# Gather the trained models to the root process
models = comm.gather(model, root=0)

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
    print(f"Testing time: {test_time:.4f} seconds")

    # Output results as Yes or No
    prediction = 'Yes' if user_pred[0] == 1 else 'No'
    print("Predicted Disease Present:", prediction)

# Finalize MPI
MPI.Finalize()

