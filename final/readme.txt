Overview:
    There are Python scripts to train models on two datasets: Open Covid Data and Covid Dataset. These
scripts are organized into folders by their respective dataset. There are provided bash scripts
to run all Python scripts for a given dataset in each folder, and the output from these scripts
will be placed in the Results folder. Information from a trial run left in results was used to create
the graphs using the notebook in the Graphing folder.


Instructions to run code:
    - These instructions are provided for a Linux environment.
    - Ensure all bash scripts have execution permissions.
    - Ensure the csv file for the Open Covid dataset has been added to final/OpenCovidData

    1. Navigate to one of the folders, CovidDataset or OpenCovidData, for the dataset you wish to run models for.
    2. Run the ./run_*.sh script in the respective folder which will output to standard output which technique's script is currently being ran.
    3. Once the bash script has finished, you can see the results in **dataset**_**technique**.txt files that have been written to the results
        with each technique in the Results folder.