from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time

start_time = time.time()
# start_time = time.process_time()

spark = SparkSession.builder.appName("SecondMostTransactions").getOrCreate()

df = spark.read.csv("House_Pricing.csv", header=True, inferSchema=True)

df = df.select('Transaction unique identifier', 'Price', 'Country')

country_counts = df.groupBy("Country").count()

country_counts = country_counts.sort(col("count").desc())

'''
country_counts = country_counts.coalesce(1)

directory_path = "2023201058_q2"

import os, shutil
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    shutil.rmtree(directory_path)    
    
country_counts.write.csv(directory_path, header=True)
'''

second_most_country = country_counts.take(2)[-1]

country_name = second_most_country["Country"]
transaction_count = second_most_country["count"]

spark.stop()

end_time = time.time()
# end_time = time.process_time()
execution_time = end_time - start_time

'''
all_files = os.listdir(directory_path)

for file in all_files:
    if file.endswith(".csv"):
        csv_file = file
        new_name = "Country_Transaction_Counts_q2.csv"
        os.rename(os.path.join(directory_path, csv_file), os.path.join(directory_path, new_name))
    else:
        os.remove(os.path.join(directory_path, file))
'''


class color:
    BLACK = '\033[1;30;48m'
    RED = '\033[1;31;48m'
    GREEN = '\033[1;32;48m'
    YELLOW = '\033[1;33;48m'
    BLUE = '\033[1;34;48m'
    PURPLE = '\033[1;35;48m'
    CYAN = '\033[1;36;48m'
    BOLD = '\033[1;37;48m'
    UNDERLINE = '\033[4;37;48m'
    END = '\033[1;37;0m'

print()
print(color.RED + "The country with the second most/highest transactions : " + country_name + color.END)
print(color.GREEN + "Number of transactions in " + country_name + " country : " + str(transaction_count) + color.END)
print(color.BLUE + "Time taken for execution in seconds : " + str(execution_time) + color.END)
print()