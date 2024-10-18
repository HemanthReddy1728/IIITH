from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time

# Start the timer
start_time = time.time()
# start_time = time.process_time()

# Create a Spark session
spark = SparkSession.builder.appName("SecondMostTransactions").getOrCreate()

# Load the dataset from the CSV file
df = spark.read.csv("House_Pricing.csv", header=True, inferSchema=True)

# Select the required columns
df = df.select('Transaction unique identifier', 'Price', 'Country')

# Group the data by 'Country' and count the number of transactions
country_counts = df.groupBy("Country").count()

# Sort the counts in descending order
country_counts = country_counts.sort(col("count").desc())
# print(country_counts.show(150))

'''
import os, shutil
# Coalesce to reduce the number of output partitions to 1
country_counts = country_counts.coalesce(1)

# Specify the output directory (not a file) where the CSV will be saved
directory_path = "2023201058_q2"

# Check if the directory exists before attempting to delete it
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    # os.rmdir(directory_path)  # Remove the directory
    shutil.rmtree(directory_path)    # Use shutil.rmtree to remove the directory and its contents

# Save the results to a CSV file (directory_path should be a directory)
country_counts.write.csv(directory_path, header=True)
'''

# Get the country with the second most transactions
second_most_country = country_counts.take(2)[-1]

# Extract the country name and transaction count
country_name = second_most_country["Country"]
transaction_count = second_most_country["count"]

# Stop the Spark session
spark.stop()

# Stop the timer
end_time = time.time()
# end_time = time.process_time()
execution_time = end_time - start_time

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
        new_name = "Country_Transaction_Counts_q2.csv"
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

# Print the result and execution time
print()
print(color.RED + "The country with the second most/highest transactions : " + country_name + color.END)
print(color.YELLOW + "Number of transactions in " + country_name + " country : " + str(transaction_count) + color.END)
print(color.CYAN + "Time taken for execution in seconds : " + str(execution_time) + color.END)
print()
