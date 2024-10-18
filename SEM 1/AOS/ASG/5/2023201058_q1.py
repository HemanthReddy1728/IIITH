from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time

# Record the start time
start_time = time.time()
# start_time = time.process_time()

# Create a Spark session
spark = SparkSession.builder.appName("SelectedCountriesHousePricing").getOrCreate()

# Load the dataset into a Spark DataFrame
data = spark.read.csv("House_Pricing.csv", header=True, inferSchema=True)

# Select the relevant columns
selected_data = data.select("Transaction unique identifier", "Price", "Country")

# Filter the data to include only the specified countries
selected_countries = selected_data.filter(selected_data["Country"].isin("GREATER LONDON", "CLEVELAND", "ESSEX"))

# Sort the data by 'Price' in descending order
selected_countries = selected_countries.orderBy(selected_countries["Price"].desc())

'''
import os, shutil
# Coalesce to reduce the number of output partitions to 1
selected_countries = selected_countries.coalesce(1)

# Specify the output directory (not a file) where the CSV will be saved
directory_path = "2023201058_q1"

# Check if the directory exists before attempting to delete it
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    # os.rmdir(directory_path)  # Remove the directory
    shutil.rmtree(directory_path)    # Use shutil.rmtree to remove the directory and its contents

# Save the results to a CSV file (directory_path should be a directory)
selected_countries.write.csv(directory_path, header=True)
'''

# Get the second highest transaction
# second_highest = selected_countries.orderBy(selected_countries["Price"].desc()).limit(2).collect()
second_highest = selected_countries.limit(2).collect()

# Extract the second highest transaction details
second_highest_transaction = second_highest[1]

# Record the end time
end_time = time.time()
# end_time = time.process_time()

# Calculate the execution time
execution_time = end_time - start_time

# Stop the Spark session
spark.stop()

'''
# Rename the CSV file and remove all other files in the folder

# List all files in the folder
all_files = os.listdir(directory_path)

# Find the CSV file and rename it
# csv_file = None
for file in all_files:
    if file.endswith(".csv"):
        csv_file = file
        # all_files.remove(csv_file)
        new_name = "Specific_Countries_Transactions_q1.csv"
        os.rename(os.path.join(directory_path, csv_file), os.path.join(directory_path, new_name))
        # break
    else:
        # Remove all other files in the folder
        os.remove(os.path.join(directory_path, file))
'''


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

# Print the second highest transaction details and execution time
print()
print(color.BLUE + "Second Highest Value Transaction among Selected Countries : " + color.END)
print(color.GREEN + "\tTransaction unique identifier : ", second_highest_transaction["Transaction unique identifier"] + color.END)
print(color.YELLOW + "\tPrice : ", str(second_highest_transaction["Price"]) + color.END)
print(color.RED + "\tCountry : ", str(second_highest_transaction["Country"]) + color.END)
print(color.CYAN + "Time taken for execution in seconds : " + str(execution_time) + color.END)
print()
