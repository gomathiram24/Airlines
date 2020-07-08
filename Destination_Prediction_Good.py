# Databricks notebook source
## Setup

# Import required libraries
import sys
import time

from tqdm import tqdm
import numpy as np
import pandas as pd

import pyspark
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import col, unix_timestamp, lit, min, max
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, LongType, DoubleType

print("System version: {}".format(sys.version))
print("Spark version: {}".format(pyspark.__version__))

# Storage account information
storage_account_name = ""
storage_account_access_key = ""
# File information 
file_location = ""
file_type = "com.databricks.spark.csv"

# Configuring the storage account
spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)
spark.conf.set("spark.databricks.io.cache.enabled", "true")

# Database setup
# %sql
# create database advanalytics2

# COMMAND ----------

#Expected percentile rank error metric function
def ROEM(predictions, userCol = "Customer", itemCol = "DestinationIndex", ratingCol = "NumOfFlights"):
  #Creates table that can be queried
  predictions.createOrReplaceTempView("predictions")
  
  #Sum of total number of flights of all destination
  denominator = predictions.groupBy().sum(ratingCol).collect()[0][0]

  #Calculating rankings of destination predictions by customer
  spark.sql("SELECT " + userCol + " , " + ratingCol + " , PERCENT_RANK() OVER (PARTITION BY " + userCol + " ORDER BY prediction DESC) AS rank FROM predictions").createOrReplaceTempView("rankings")

  #Multiplies the rank of each destination by the number of flights and adds the products together
  numerator= spark.sql('SELECT SUM(' + ratingCol + ' * rank) FROM rankings').collect()[0][0]
  performance = numerator/denominator
  
  return performance

# COMMAND ----------

# Load transformed data
data_path = ""
customer_data = spark.read.format("delta").load(data_path).orderBy(functions.rand(), seed=0)
customer_data = customer_data.filter(customer_data["NumOfFlights"] > 0)

# Build train and validation set
train_frac = 0.7
(train_data, validation_data) = customer_data.randomSplit([train_frac, 1-train_frac], seed=0)

# COMMAND ----------

display(validation_data)

# COMMAND ----------

# Hyperparameter Tuning
# Empty list to be filled with models
model_list = []

# Complete each of the hyperparameter value lists
ranks = [10]
maxIters = [10]
regParams = [1.0]
alphas = [1.0]

# For loop will automatically create and store ALS models
for r in ranks:
  for mi in maxIters:
    for rp in regParams:
      for a in alphas:
        model_list.append(ALS(userCol = "Customer", itemCol = "DestinationIndex",
                             ratingCol = "NumOfFlights", rank = r, maxIter = mi, 
                              regParam = rp, alpha = a, coldStartStrategy = "drop", 
                             nonnegative = True, implicitPrefs = True))

# COMMAND ----------

# Training and selecting best model

# Empty list to fill with ROEMs from each model
best_score = 10**12
best_fitted_model = None

# Loops through all models and all folds
for model in tqdm(model_list):
  # Fits model to all of training data and generates preds for test data
  fitted_model = model.fit(train_data)
  predictions = fitted_model.transform(validation_data)
  v_ROEM = ROEM(predictions)
  
  print ("Validation ROEM: ", v_ROEM)
  if v_ROEM == 0:
    continue
  if v_ROEM < best_score:
    best_score, best_fitted_model =  v_ROEM, fitted_model
    
    #save best fitted model

# COMMAND ----------

i = np.argmin(best_score)
print(i)
print(best_score)

# COMMAND ----------

best_model = model_list[i]
print(best_model)

# COMMAND ----------

# Extract the Rank
best_rank = best_model.getRank()
print("Rank: ", best_rank)

# Extract the MaxIter value
best_MaxIter = best_model.getMaxIter()
print("MaxIter: ", best_MaxIter)

# Extract the RegParam value
best_RegParam = best_model.getRegParam()
print("RegParam: ", best_RegParam)

# Extract the Alpha value
best_Alpha = best_model.getAlpha()
print("Alpha: ", best_Alpha)

# COMMAND ----------

# save and load the model
#best_fitted_model.save("wasbs://temp@jbspoteastusdevsa.blob.core.windows.net/best_fitted_model_1_yr")
#it best_fitted_model = ALSModel.load("temp@jbspoteastusdevsa.blob.core.windows.net/Dest_Pred_best_fitted_model_1.1")

# COMMAND ----------

# Getting recommendations
custRecs = best_fitted_model.recommendForAllUsers(3)
custRecs.show(10)

# COMMAND ----------

# Decoder Dict
unique_values = spark.read.format("csv").load("",  format="com.databricks.spark.csv", 
                      header="true")
unique_values= unique_values.withColumn("DestinationIndex", unique_values["DestinationIndex"].cast(DoubleType()))
uni_dic = unique_values.toPandas().set_index('DestinationIndex').T.to_dict('Destinations')

# COMMAND ----------

# Loading Top destination data
Top_Des = spark.read.format("csv").load("",  format="com.databricks.spark.csv", 
                      header="true")
top_des = [row.Destinations for row in Top_Des.collect()]

# COMMAND ----------

#count of top destinations
c_JFK, c_BOS , c_FLL = 0 , 0 , 0
for i in custRecs.collect():
  Rec1 = (list(uni_dic.get(i[1][0][0]).values())[0]).strip()
  Rec2 = (list(uni_dic.get(i[1][1][0]).values())[0]).strip()
  Rec3 = (list(uni_dic.get(i[1][2][0]).values())[0]).strip()
  if top_des[0] in [Rec1, Rec2, Rec3]:
    c_JFK+=1
  if top_des[1] in [Rec1, Rec2, Rec3]:
    c_BOS+=1
  if top_des[2] in [Rec1, Rec2, Rec3]:
    c_FLL+=1
print('JFK' , c_JFK , 'BOS' , c_BOS , 'FLL' , c_FLL)

# COMMAND ----------

# saving results in dataframe
REC = []
for i in custRecs.collect():
  Rec1 = (list(uni_dic.get(i[1][0][0]).values())[0]).strip()
  Rec2 = (list(uni_dic.get(i[1][1][0]).values())[0]).strip()
  Rec3 = (list(uni_dic.get(i[1][2][0]).values())[0]).strip()
  REC.append([i[0], Rec1, Rec2, Rec3])
print(REC)
Rec_df = pd.DataFrame(REC, columns = ['Customer' , 'Rec1', 'Rec2', 'Rec3'])

# COMMAND ----------


#count top_dest
c_JFK, c_BOS , c_FLL = 0 , 0 , 0
for i in custRecs.collect():
  recs = []
  for idx in range(3):
    recs.extend(rec.strip() for rec in uni_dic.get(i[1][idx][0], {}).values())
    if top_des[0] in recs:
      c_JFK+=1
    if top_des[1] in recs:
      c_BOS+=1
    if top_des[2] in recs:
      c_FLL+=1
print('JFK' , c_JFK , 'BOS' , c_BOS , 'FLL' , c_FLL)

# COMMAND ----------

#Convert pandas DF to spark df
df = spark.createDataFrame(Rec_df, schema=None)

# COMMAND ----------

#Display
display(df)

# COMMAND ----------


