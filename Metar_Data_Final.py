# Databricks notebook source
# MAGIC %md ###DATA COLLECTION 

# COMMAND ----------

# Import libraries
from __future__ import print_function
import datetime
from urllib.request import urlopen
import pandas as pd

# COMMAND ----------

# Loading ICAO Airport code
# File location and type
file_location = "/FileStore/tables/ICAO_Code.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Selecting  airport column from df
df1 = df.select("_c4")
df1 = df1.withColumnRenamed("_c4", "stations")
df1 = df1.toPandas()
airports = df1.to_dict("records")

# COMMAND ----------

# input url
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"

# Number of attempts to download data
MAX_ATTEMPTS = 2

# COMMAND ----------

def download_data(uri):
    """Fetch the data from the IEM
    The IEM download service has some protections in place to keep the number
    of inbound requests in check.  This function implements an exponential
    backoff to keep individual downloads from erroring
    Args:
      uri (string): URL to fetch
    Returns:
      string data
    """
    attempt = 0
    while attempt <= MAX_ATTEMPTS:
        try:
            data = urlopen(uri, timeout=300).read().decode("utf-8")
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as exp:
            print("download_data(%s) failed with %s" % (uri, exp))
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return data

# COMMAND ----------

def get_stations_from_filelist(airports):
    """Build a listing of stations from a simple file listing the stations.
    The file should simply have one station per line.
    """
    station = []
    for airport in airports:
      station.append(airport["stations"])
    return station

# COMMAND ----------

def main():
    """main method saves csv file for all jetblue airports for given dates"""
    START_DT = pd.datetime.now() - datetime.timedelta(hours=1)
    END_DT= pd.datetime.now() - datetime.timedelta(hours=1)

    service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"

    service += START_DT.strftime("year1=%Y&month1=%m&day1=%d&hour1=%H&")
    service += END_DT.strftime("year2=%Y&month2=%m&day2=%d&hour2=%H&")


    stations = get_stations_from_filelist(airports)
    
   
    for station in stations:
        uri = "%s&station=%s" % (service, station)
        data = download_data(uri)
        dbutils.fs.put("dbfs:/_METAR_13_/{}_{}_{}.csv".format(station, START_DT.strftime("%Y_%m_%d_%H_%M_%S"), END_DT.strftime("%Y_%m_%d_%H_%M_%S")), data)

# COMMAND ----------

if __name__ == "__main__":
    main()

# COMMAND ----------

# MAGIC %md ###LOADING DATA

# COMMAND ----------

# Importing multiple csv files from saved folder and concatenating in to one DataFrame
path = 'dbfs:/_METAR_13_/*.csv'
df = spark.read.format("csv").option("header", "true").option("inferSchema", "false").option("comment", "#").load(path)

# COMMAND ----------

START_DT = pd.datetime.now() - datetime.timedelta(hours=1)

# COMMAND ----------

START_DT

# COMMAND ----------

display(df)

# COMMAND ----------

df1 = df.filter(df.valid > START_DT.strftime("%Y-%m-%d %H:%M"))

# COMMAND ----------

display(df1)

# COMMAND ----------

# Saving transformed data in to blob
# Storage account information
storage_account_name = 
storage_account_access_key = 


# Configuring the storage account
spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)
spark.conf.set("spark.databricks.io.cache.enabled", "true")


# COMMAND ----------

# Saving transformed data in to blob
# DONT RUN THIS AGAIN
# creating transformed data
df1.write.format("csv").option("path","wasbs://metar-cont@.blob.core.windows.net/{}/{}/{}.csv".format(START_DT.strftime("%Y"), START_DT.strftime("%m"), START_DT.strftime("%d_%H_%M_%S"))).saveAsTable("METAR.{}".format(START_DT.strftime("%Y_%m_%d_%H_%M_%S")))

# COMMAND ----------

# Saving transformed data in to blob
# Storage account information
storage_account_name = 
storage_account_access_key = 


# Configuring the storage account
spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)
spark.conf.set("spark.databricks.io.cache.enabled", "true")






