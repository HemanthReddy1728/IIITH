from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time

# Start the timer
start_time = time.time()
# start_time = time.process_time()

# Create a SparkSession
spark = SparkSession.builder.appName("AllCountriesTransactions").getOrCreate()

# Load the dataset into a PySpark DataFrame
df = spark.read.csv("House_Pricing.csv", header=True, inferSchema=True)

# Select the required columns
df = df.select('Transaction unique identifier', 'Price', 'Country')

# Group the data by 'Country' and count the number of transactions for each country
country_counts = df.groupBy('Country').count()


import os, shutil
# Coalesce to reduce the number of output partitions to 1
country_counts = country_counts.coalesce(1)

# Specify the output directory (not a file) where the CSV will be saved
directory_path = "2023201058_q3"

# Check if the directory exists before attempting to delete it
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    # os.rmdir(directory_path)  # Remove the directory
    shutil.rmtree(directory_path)    # Use shutil.rmtree to remove the directory and its contents

# Save the results to a CSV file (directory_path should be a directory)
country_counts.write.csv(directory_path, header=True)


# Stop the Spark session
spark.stop()


# Rename the CSV file and remove all other files in the folder

# List all files in the folder
all_files = os.listdir(directory_path)

# Find the CSV file and rename it
# csv_file = None
for file in all_files:
    if file.endswith(".csv"):
        csv_file = file
        # all_files.remove(csv_file)
        new_name = "Country_Transaction_Counts_q3.csv"
        os.rename(os.path.join(directory_path, csv_file), os.path.join(directory_path, new_name))
        # break
    else:
        # Remove all other files in the folder
        os.remove(os.path.join(directory_path, file))


# Calculate the time taken for execution
end_time = time.time()
# end_time = time.process_time()
execution_time = end_time - start_time

class color:
    PURPLE = '\033[1;35;48m'
    CYAN = '\033[1;36;48m'
    BOLD = '\033[1;37;48m'
    BLUE = '\033[1;34;48m'
    GREEN = '\033[1;32;48m'
    YELLOW = '\033[1;33;48m'
    RED = '\033[1;31;48m'
    BLACK = '\033[1;30;48m'
    UNDERLINE = '\033[4;37;48m'
    END = '\033[1;37;0m'


# Print the execution time
print()
print(color.RED + "A CSV file of name " + new_name  + " is created in " +  directory_path + " folder." + color.END)
print(color.YELLOW + "The above CSV file contains Number of transactions per Country" + color.END)
print(color.CYAN + "Time taken for execution in seconds : " + str(execution_time) + color.END)
print()
