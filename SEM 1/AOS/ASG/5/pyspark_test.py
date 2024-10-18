# import sys
# import pyspark

from pyspark.sql.functions import *
from pyspark.sql import SparkSession

sc = SparkSession.builder.appName("Test")
sc = sc.getOrCreate()

data_csv = sc.read.csv('airports.csv', header=True)

grouped_data = data_csv.groupBy("COUNTRY").count().sort(desc("count"))

output1 = grouped_data.select("COUNTRY").first()

# data_csv = sc.read.csv('House_Pricing.csv', header=True)

# grouped_data = data_csv.groupBy("Country").count().sort(desc("count"))

# output1 = grouped_data.select("Country").first()

ans = output1[0]
print(ans)

sc.stop()