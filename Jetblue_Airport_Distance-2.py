# Databricks notebook source
import pandas as pd
import math

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
df_co_ord = spark.read.format(file_type).option("header", True).option("delimiter", ",").option("inferSchema", "true").load(file_location)

df_co_ord = df_co_ord.filter(df_co_ord.JBLUAirport=='Y')

df_co_ord = df_co_ord.select("AirportCode", "AirportGPSLAT", "AirportGPSLONG")

display(df_co_ord)

# COMMAND ----------

df_co_ord.count()

# COMMAND ----------

df_nc= spark.sql("""select * from nc_pilots_csv""")
display(df_nc)

# COMMAND ----------

df_nc.count()

# COMMAND ----------

df_nc_a1 = df_nc.select("BidBase")
df_nc_a1 = df_nc_a1.join(df_co_ord, df_co_ord.AirportCode==df_nc_a1.BidBase, "inner").drop(df_co_ord.AirportCode)
display(df_nc_a1)

# COMMAND ----------

df_nc_a1 = df_nc_a1.distinct()
display(df_nc_a1)

# COMMAND ----------

df_nc_a2 = df_nc.select("NearestAirport")
df_nc_a2 = df_nc_a2.join(df_co_ord, df_co_ord.AirportCode==df_nc_a2.NearestAirport, "left").drop(df_co_ord.AirportCode)
display(df_nc_a2)

# COMMAND ----------

df_nc_a2.count()

# COMMAND ----------

df_nc_a1 = df_nc_a1.toPandas()
airports_1 = df_nc_a1.to_dict("records")

# COMMAND ----------

for a in airports_1:
  print(a)
  break

# COMMAND ----------

df_nc_a2 = df_nc_a2.toPandas()
airports_2 = df_nc_a2.to_dict("records")

# COMMAND ----------

def calculate_haversine(airports_2, airports_2):
    Data = []
    ''' Function that calculates the great-circle (Haversine) distance between 2 locations in miles'''
    for airport in airports_1:
      for airports in airports_2:
        data = []
        lat_1 = airport["AirportGPSLAT"]
        lon_1 = airport["AirportGPSLONG"]
        lat_2 = airports["AirportGPSLAT"]
        lon_2 = airports["AirportGPSLONG"]        
        R = 6373.0 # radius of the earth
        lat1 = math.radians(lat_1)
        lon1 = math.radians(lon_1)
        lat2 = math.radians(lat_2)
        lon2 = math.radians(lon_2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        distance = distance * 0.621371 # convert to miles
        data.append(airport["BidBase"])
        data.append(airports["NearestAirport"])
        data.append(distance)
        Data.append(data)
    df = pd.DataFrame(Data, columns = ['BidBase', 'NearestAirport', 'Distance'])
    df1 = spark.createDataFrame(df)
    df1.write.saveAsTable("raw_crew_data" + "." + "distance", mode= "append")
    return display(df1)

# COMMAND ----------

calculate_haversine(airports_1, airports_2)

# COMMAND ----------


