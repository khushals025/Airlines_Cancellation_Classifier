# Databricks notebook source
# MAGIC %fs mkdirs dbfs:/AirlinesProject/OutputData

# COMMAND ----------

# MAGIC %fs mkdirs dbfs:/AirlinesProject/OutputData/Merged_Data

# COMMAND ----------

# MAGIC %fs ls dbfs:/AirlinesProject/

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

# Initialize a Spark session
spark = SparkSession.builder.appName("AddBinaryColumns").getOrCreate()

# Load your Parquet file as a DataFrame (replace 'your_parquet_path' with your actual Parquet file path)
df = spark.read.parquet("/AirlinesProject/*/*/*.parquet")

# Add new columns for arrival and departure delay indicators
df = df.withColumn("arrival_delay_indicator", when(col("arrival_delay") > 0, "yes").otherwise("no"))
df = df.withColumn("departure_delay_indicator", when(col("departure_delay") > 0, "yes").otherwise("no"))

# Show the updated DataFrame with the new columns

df = df.repartition(20)
df.write.csv("/AirlinesProject/OutputData/Merged_Data/", header=True, mode="overwrite")

# COMMAND ----------

# Stop the Spark session
spark.stop()