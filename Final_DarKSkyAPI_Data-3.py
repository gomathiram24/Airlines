# Databricks notebook source
# importing libraries
from datetime import datetime
import json
import pandas as pd
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType, TimestampType
import requests
from tqdm import tqdm

# COMMAND ----------

# Storage info for saving the data to the blob

# Storage account information
storage_account_name = ""
storage_account_access_key = ""

# Configuring the storage account
spark.conf.set(
    "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
    storage_account_access_key)

# COMMAND ----------

# Loading Airport co-ordinates
# Blob storage account information
storage_account_name = ""
storage_account_access_key = ""

# Blob directory and file type for vw_flights_fact files
file_location = ""
file_type = "csv"

# Configure the blob storage account
spark.conf.set(
    "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
    storage_account_access_key)

# Create a dataframe from the blob files
df = spark.read.format(file_type).option("header", True).option(
    "delimiter", ",").option("inferSchema", "true").load(file_location)

# Data Transformation
df1 = df.filter(df.JBLUAirport == "Y")
df2 = df1.select("AirportGPSLAT", "AirportGPSLONG", "AirportCode")
df3 = df2.distinct()
df3 = df3.withColumnRenamed("AirportGPSLAT", "lat")
df4 = df3.withColumnRenamed("AirportGPSLONG", "long")
df4 = df4.withColumnRenamed("AirportCode", "airport_cd")
df5 = df4.toPandas()
airports = df5.to_dict("records")

# COMMAND ----------

# input dates
START_DT = "1/7/2018"
END_DT = "1/12/2018"
dates = pd.date_range(START_DT, END_DT)

# COMMAND ----------


def fetch_weather_json(lat, long, airport_cd, date):
    """loads data if it exits in dbfs else collecting raw data from API """
    """path is different while saving and loading"""
    try:
        with open("/dbfs/sample/{}_{}.json".format(airport_cd, date.strftime("%Y-%m-%dT00-00-00-00")), "r") as read_file:
            data = json.load(read_file)
            print("Loading from disk ...")
    except:
        print("Getting from API...")
        print(path)
        response = requests.get("https://api.darksky.net/forecast/8ed965709ce9ad9bc0e81400572af37a/{},{},{}".format(
            lat, long, date.strftime("%Y-%m-%dT00:00:00-0400")), "hourly")
        data = response.json()
        dbutils.fs.put("dbfs:/sample/{}_{}.json".format(airport_cd,
                                                        date.strftime("%Y-%m-%dT00-00-00-00")), json.dumps(data))
    return data

# COMMAND ----------


def get_weather_data(dates, airports):
    """extracting hourly data from raw json and saving it in pandas df"""
    hour_data = []
    for date in dates:
        for airport in airports:
            raw_data = fetch_weather_json(
                airport["lat"], airport["long"], airport["airport_cd"], date)
            try:
                for i, j in enumerate(raw_data["hourly"]["data"]):
                    j["latitude"] = airport["lat"]
                    j["longitude"] = airport["long"]
                    j["request_date"] = date
                    j["airport_code"] = airport["airport_cd"]
                    hour_data.append(j)
                    # only keep first 24 hours of forecasted data (48 hours provided)
                    if i == 23:
                        break
            except KeyError:
                pass
    df_hourly = pd.DataFrame(hour_data)
    return df_hourly

# COMMAND ----------


df = get_weather_data(dates, airports)

# COMMAND ----------

df.head()

# COMMAND ----------
