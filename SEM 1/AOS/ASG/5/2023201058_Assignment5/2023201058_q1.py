from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time

start_time = time.time()
# start_time = time.process_time()

spark = SparkSession.builder.appName("SelectedCountriesHousePricing").getOrCreate()

data = spark.read.csv("House_Pricing.csv", header=True, inferSchema=True)

selected_data = data.select("Transaction unique identifier", "Price", "Country")

selected_countries = selected_data.filter(selected_data["Country"].isin("GREATER LONDON", "CLEVELAND", "ESSEX"))

selected_countries = selected_countries.orderBy(selected_countries["Price"].desc())

'''
selected_countries = selected_countries.coalesce(1)

directory_path = "2023201058_q1"

import os, shutil
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    shutil.rmtree(directory_path)    

selected_countries.write.csv(directory_path, header=True)
'''

# second_highest = selected_countries.orderBy(selected_countries["Price"].desc()).limit(2).collect()
second_highest = selected_countries.limit(2).collect()

second_highest_transaction = second_highest[1]

spark.stop()

end_time = time.time()
# end_time = time.process_time()
execution_time = end_time - start_time

'''
all_files = os.listdir(directory_path)

for file in all_files:
    if file.endswith(".csv"):
        csv_file = file
        new_name = "Specific_Countries_Transactions_q1.csv"
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
print(color.BLUE + "Second Highest Value Transaction among Selected Countries : " + color.END)
print(color.RED + "\tTransaction unique identifier : ", second_highest_transaction["Transaction unique identifier"] + color.END)
print(color.GREEN + "\tPrice : ", str(second_highest_transaction["Price"]) + color.END)
print(color.PURPLE + "\tCountry : ", str(second_highest_transaction["Country"]) + color.END)
print(color.CYAN + "Time taken for execution in seconds : " + str(execution_time) + color.END)
print()
