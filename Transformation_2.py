# Databricks notebook source
# MAGIC %fs mkdirs dbfs:/AirlinesProject/OutputData/group_by_month

# COMMAND ----------

# MAGIC %fs mkdirs dbfs:/AirlinesProject/OutputData/group_by_carrier

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, when, col
count = 0


# Initialize a Spark session
spark = SparkSession.builder.appName("DelaysAnalysis").getOrCreate()

# Load your Parquet file as a DataFrame (replace 'your_parquet_path' with your actual Parquet file path)
paths = ["/AirlinesProject/Year=2005","/AirlinesProject/Year=2006", "/AirlinesProject/Year=2007", "/AirlinesProject/Year=2008" ]

for i in paths:
    df = spark.read.parquet(i)
   
    if i == paths[0]:
        custom_file_name = "Year=2005"
    elif i == paths[1]:
        custom_file_name = "Year=2006"
    elif i == paths[2]:
        custom_file_name = 'Year=2007'
    else:
        custom_file_name = "Year=2008"


# Create a table view to perform SQL operations
    df.createOrReplaceTempView("flight_data")

# Table 1: Number of arrival delays and departure delays per unique carrier
    carrier_delays_df = spark.sql("""
        SELECT unique_carrier,
              SUM(CASE WHEN arrival_delay > 0 THEN 1 ELSE 0 END) AS arrival_delays,
              SUM(CASE WHEN departure_delay > 0 THEN 1 ELSE 0 END) AS departure_delays
        FROM flight_data
        GROUP BY unique_carrier
        ORDER BY unique_carrier
    """)

# Show the result for the first table
    carrier_delays_df.show()
    carrier_delays_df.write.csv(f"/AirlinesProject/OutputData/group_by_carrier/{custom_file_name}", header = True, mode = "overwrite")

# Table 2: Number of arrival delays and departure delays per month
    month_delays_df = spark.sql("""
        SELECT month,
              SUM(CASE WHEN arrival_delay > 0 THEN 1 ELSE 0 END) AS arrival_delays,
              SUM(CASE WHEN departure_delay > 0 THEN 1 ELSE 0 END) AS departure_delays
        FROM flight_data
        GROUP BY month
        ORDER BY month
    """)

# Show the result for the second table
    month_delays_df.show()
    month_delays_df.write.csv(f"/AirlinesProject/OutputData/group_by_month/{custom_file_name}", header = True, mode = "overwrite")
    
# Stop the Spark session

spark_version = spark.version
print("Spark Version:", spark_version)



# COMMAND ----------

spark.stop()