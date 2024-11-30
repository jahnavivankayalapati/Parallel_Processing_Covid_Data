==================================================================================
Finalcodedataset1.py: COVID Data Analysis and Modeling Using Parallel Techniques
==================================================================================
==========
Overview:
==========
This program performs parallel training and testing of a Random Forest model using MPI (Message Passing Interface) and various parallelization strategies. It compares the performance of the following methods:
1. Multi-Processing
2. MPI 
3. SPMD (Single Program Multiple Data)
4. MPMD (Multiple Programs Multiple Data)

==============
Prerequisites:
==============
Ensure the following dependencies are installed:
1. Python (Version >= 3.7)
2. MPI Implementation 
3. Required Python libraries:
4. numpy
5. pandas
6. scikit-learn
7. mpi4py

======
Setup:
======
1. Download the Dataset: Covid Dataset.csv
=> Ensure the file Covid Dataset.csv is in the same directory as the program.
=> The dataset should be a CSV file with a column named COVID-19 for labels.

2. Download the Code file: Finalcodedataset1.py
=>Clone the project from the GitHub repository:
=>Repository URL: https://github.com/jahnavivankayalapati/Parallel_Processing_Covid_Data
=>Navigate to the project directory after cloning.

3. Run the Program:
=> Ensure the script Finalcodedataset1.py is present in the directory with the dataset.
=>command: python Finalcodedataset1.py