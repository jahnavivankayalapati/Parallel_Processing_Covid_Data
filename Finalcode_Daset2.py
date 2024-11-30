import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from mpi4py import MPI
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
import concurrent.futures
import time

# Load and preprocess the data
main_data = pd.read_csv('owid-covid-data.csv')

# Sampling the dataset to 10,000 rows
main_data = main_data.sample(10000, random_state=42)

# Handle missing values by filling them with the median
main_data.fillna(main_data.median(numeric_only=True), inplace=True)

# Select preferred features and the target variable
features = [
    'total_cases', 'total_deaths', 'total_tests', 'total_vaccinations',
    'reproduction_rate', 'icu_patients', 'hosp_patients',
    'population_density', 'gdp_per_capita', 'life_expectancy'
]
target = 'new_cases'

# Filter the dataset
X = main_data[features]
y = main_data[target]

# Encode categorical variables if any exist in the selected features
label_encoder = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = label_encoder.fit_transform(X[col])

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Example input for prediction
example_input = pd.DataFrame({
    'total_cases': [1_000_000],
    'total_deaths': [20_000],
    'total_tests': [2_000_000],
    'total_vaccinations': [500_000],
    'reproduction_rate': [1.2],
    'icu_patients': [100],
    'hosp_patients': [50000],
    'population_density': [3000],
    'gdp_per_capita': [50_000],
    'life_expectancy': [80]
})
example_input = scaler.transform(example_input)

# 1. Standard Random Forest
print("### Standard Random Forest ###")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

start_time = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start_time

start_time = time.time()
y_pred = rf.predict(X_test)
test_time = time.time() - start_time

example_prediction = int(rf.predict(example_input)[0])
accuracy = r2_score(y_test, y_pred) * 100

print(f"Training Time: {train_time:.4f} seconds")
print(f"Testing Time: {test_time:.4f} seconds")
print(f"Example Input Prediction: {example_prediction}")
print(f"Accuracy: {accuracy:.2f}%\n")

# 2. Random Forest with Multiprocessing
print("### Random Forest with Multiprocessing ###")
num_workers = 4
data_chunks = np.array_split(X_train, num_workers)
target_chunks = np.array_split(y_train, num_workers)

def train_rf_chunk(chunk):
    X_chunk, y_chunk = chunk
    rf_chunk = RandomForestRegressor(n_estimators=25, max_depth=10, random_state=42)
    rf_chunk.fit(X_chunk, y_chunk)
    return rf_chunk

start_time = time.time()
futures = []
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for X_chunk, y_chunk in zip(data_chunks, target_chunks):
        futures.append(executor.submit(train_rf_chunk, (X_chunk, y_chunk)))

models = [future.result() for future in concurrent.futures.as_completed(futures)]
train_time = time.time() - start_time

start_time = time.time()
y_pred = np.mean([model.predict(X_test) for model in models], axis=0)
test_time = time.time() - start_time

example_prediction = int(np.mean([model.predict(example_input) for model in models], axis=0))
accuracy = r2_score(y_test, y_pred) * 100

print(f"Training Time: {train_time:.4f} seconds")
print(f"Testing Time: {test_time:.4f} seconds")
print(f"Example Input Prediction: {example_prediction}")
print(f"Accuracy: {accuracy:.2f}%\n")

# 3. Optimized Random Forest with MPI
print("### Optimized Random Forest with MPI ###")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size > 4:
    raise ValueError("This script is optimized for up to 4 MPI workers.")

# Split data among MPI workers
X_split = np.array_split(X_train, size)
y_split = np.array_split(y_train, size)

X_local = comm.scatter(X_split, root=0)
y_local = comm.scatter(y_split, root=0)

local_rf = RandomForestRegressor(
    n_estimators=50 // size,
    max_depth=8,
    min_samples_split=5,
    random_state=42
)
start_time = time.time()
local_rf.fit(X_local, y_local)
local_train_time = time.time() - start_time

models = comm.gather(local_rf, root=0)

if rank == 0:
    start_time = time.time()
    y_pred = np.mean([model.predict(X_test) for model in models], axis=0)
    test_time = time.time() - start_time

    example_prediction = int(np.mean([model.predict(example_input) for model in models], axis=0))
    accuracy = r2_score(y_test, y_pred) * 100

    print(f"Training Time: {local_train_time:.4f} seconds")
    print(f"Testing Time: {test_time:.4f} seconds")
    print(f"Example Input Prediction: {example_prediction}")
    print(f"Accuracy: {accuracy:.2f}%\n")
