# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, LongType, TimestampType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sbs
from matplotlib.ticker import MaxNLocator
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StandardScaler
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import Normalizer
from pyspark.sql.functions import col, skewness, kurtosis

# COMMAND ----------

# MAGIC %md ### DATA EXTRACTION

# COMMAND ----------

# Filtering customers who had at least one flight during 21 days window before signing up for credit card

df = spark.sql(
    """select * 
                  from raw_dwjetblue2.rawdata_tktcoupon 
                  join export_comarch.export_all_2019_customers 
                  on rawdata_tktcoupon.FrequentTravelerNbr=export_all_2019_customers.TrueBlueNumber
                  where 
                  cast(export_all_2019_customers.CardOpenDate as date) >= '2019-01-01'
                  and
                  cast(rawdata_tktcoupon.ServiceStartDate as date)
                  between 
                  date_sub(cast(export_all_2019_customers.CardOpenDate as date), 21) and cast(export_all_2019_customers.CardOpenDate as date)
                  and
                  raw_dwjetblue2.rawdata_tktcoupon.CouponStatus == 'USED'
                  and
                  raw_dwjetblue2.rawdata_tktcoupon.PreviousCouponStatusCodeBYTE == 'LFTD'
                  """
)

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install black

# COMMAND ----------


def data_ext(df):
    """ Calculating day delta between CardOpenDate & ServiceStartDate as DBC(Days Before Card signup) and 
      ServiceStartDate & PNRCreateDate as DBD(Days Before Departure)
      Applying window function to select the recent flight by filtering roundtrips and layovers
  """
    df_delta = df.withColumn("DBC", datediff("CardOpenDate", "ServiceStartDate"))
    df_delta = df_delta.withColumn("DBD", datediff("ServiceStartDate", "PNRCreateDate"))
    df_win = df_delta.withColumn(
        "rank",
        rank().over(
            Window.partitionBy("TrueBlueNumber").orderBy(desc("ServiceStartDate"))
        ),
    )
    df_least = df_win.filter(df_win.rank == "1")
    return df_least


# COMMAND ----------

df_least = data_ext(df)
display(df_least)

# COMMAND ----------

# MAGIC %md ### FEATURE SELECTION/TRANSFORMATION

# COMMAND ----------

# Removing outliers by trimming the data
age = 110
least_fare_amount = 1
most_fare_amount = 600
DBD = 100

# COMMAND ----------


def feature_selection(df_least):
    """ Selecting features and converting datatypes to Integer and Double
      Trimming data to remove outliers
  """
    # TODO: Don't hardcode dates

    df_selection = (
        data_ext(df)
        .select(
            "CurrentPointsBalance",
            "LifetimePointsEarned",
            "FareBreakAmt",
            "DBC",
            "DBD",
            "Age",
            "Is2019Mosaic",
            "Gender",
        )
        .withColumn(
            "CurrentPointsBalance", col("CurrentPointsBalance").cast(IntegerType())
        )
        .withColumn(
            "LifetimePointsEarned", col("LifetimePointsEarned").cast(IntegerType())
        )
        .withColumn("Age", col("Age").cast(IntegerType()))
        .withColumn("FareBreakAmt", col("FareBreakAmt").cast(DoubleType()))
        .filter(col("Age") < age)
        .filter(
            (col("FareBreakAmt") > least_fare_amount)
            & (col("FareBreakAmt") < most_fare_amount)
        )
        .filter(col("DBD") < DBD)
    )
    return df_selection


# COMMAND ----------

df_selection = feature_selection(df_least)
display(df_selection)

# COMMAND ----------

# Applying udf for mosaic indicator and gender classification and age classification


def data_transformation(df_selection):
    """ Flagging Mosaic members and Genders as 1 or 0
      Grouping Age in to three categories : Millenial-1, Middle Age-2, Senior Age-3
      Flagging Grouped categories as 1 or 0
      Transforming all the features in to vector column
      Normalizing vector column
  """

    def trans(df_selection):
        df_selection = df_selection.withColumn(
            "Is2019Mosaic_index", when(col("Is2019Mosaic") == "Y", 1).otherwise(0)
        )
        df_selection = df_selection.withColumn(
            "Gender_index", when(col("Gender") == "M", 1).otherwise(0)
        )
        df_selection = df_selection.withColumn(
            "Age_Index",
            when((col("Age") > 21) & (col("Age") <= 35), 1)
            .when((col("Age") > 35) & (col("Age") <= 50), 2)
            .when(col("Age") > 50, 3)
            .otherwise(0),
        )
        return df_selection

    df_vec = trans(df_selection)

    columns = [
        "CurrentPointsBalance_Jan_22_2020",
        "LifetimePointsEarned",
        "Is2019Mosaic_index",
        "Gender_index",
        "DBC",
        "DBD",
        "FareBreakAmt",
        "Age_Index",
    ]
    vecAssembler = VectorAssembler(inputCols=columns, outputCol="features")
    vec_dataset = (
        vecAssembler.setHandleInvalid("skip").transform(df_vec).select("features")
    )
    scaler = StandardScaler(inputCol="features", outputCol="ScaledFeatures")
    scalerModel = scaler.fit(vec_dataset.select("features"))
    scaled_dataset = scalerModel.transform(vec_dataset).select("ScaledFeatures")
    dataset = scaled_dataset.withColumnRenamed("ScaledFeatures", "features")
    return dataset


# COMMAND ----------

df_model = data_transformation(df_selection)

# COMMAND ----------

# MAGIC %md ###Elbow Method

# COMMAND ----------


def elbow(dataset):
    """
  """
    cost = np.zeros(20)
    for k in range(2, 20):
        kmeans = KMeans().setK(k).setSeed(1)
        model = kmeans.fit(dataset)
        cost[k] = model.computeCost(dataset)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, 20), cost[2:20])
    ax.set_xlabel("k")
    ax.set_ylabel("cost")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return display(ax)


# COMMAND ----------

elbow(df_model)

# COMMAND ----------

# MAGIC %md ###Silhouette Analysis

# COMMAND ----------

# Determine optimal number of clusters by using Silhoutte Score Analysis
def optimal_k(dataset, k_min, k_max, num_runs):
    """Determine optimal number of clusters by using Silhoutte Score Analysis.
    df_in: the input dataframe
    k_min: the minmum number of the clusters
    k_max: the maxmum number of the clusters
    num_runs: the number of runs for each fixed clusters
    k: optimal number of the clusters which will be used to train the final model
    silh_list: Silhouette score
    r_table: the running results table
    
  """
    silh_list = []
    k_list = np.arange(k_min, k_max + 1)
    r_table = dataset.select("features").toPandas()
    centers = pd.DataFrame()
    for k in k_list:
        silh_val = []
        for run in np.arange(1, num_runs + 1):

            # Trains a k-means model.
            kmeans = KMeans().setK(k).setSeed(int(np.random.randint(100, size=1)))
            model = kmeans.fit(dataset)

            # Make predictions
            predictions = model.transform(dataset)
            r_table["cluster_{k}_{run}".format(k=k, run=run)] = predictions.select(
                "prediction"
            ).toPandas()

            # Evaluate clustering by computing Silhouette score
            evaluator = ClusteringEvaluator()
            silhouette = evaluator.evaluate(predictions)
            silh_val.append(silhouette)

        silh_array = np.asanyarray(silh_val)
        silh_list.append(silh_array.mean())

    silhouette = pd.DataFrame(list(zip(k_list, silh_list)), columns=["k", "silhouette"])
    return k_list[np.argmax(silh_list, axis=0)], silhouette, r_table


# COMMAND ----------

# MAGIC %md ###FINAL

# COMMAND ----------


def model(dataset):
    """ Trains model with optimal k from optimal k function
      Returns Silhoutte score for the model
      Returns the cluster centers 
  """
    k, silh_list, r_table = optimal_k(dataset, 2, 10, 5)
    spark.createDataFrame(silh_list).show()
    kmeans = KMeans().setK(k).setSeed(1)
    result_table = dataset.select("features").toPandas()
    model = kmeans.fit(dataset)
    predictions = model.transform(dataset)
    result_table["k"] = predictions.select("prediction").toPandas()
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    ctr = []
    for center in centers:
        ctr.append(center)
        print(center)
    out_features = [
        "CurrentPointsBalance_Jan_22_2020",
        "LifetimePointsEarned",
        "Is2019Mosaic_index",
        "Gender_index",
        "DBC",
        "DBD",
        "FareBreakAmt",
        "Age_Index",
    ]
    predictions.groupBy("prediction").count().show()
    centers = pd.DataFrame(ctr, columns=out_features)
    return centers.head(), result_table


# COMMAND ----------

centers, result = model(df_model)

# COMMAND ----------

centers.head()

# COMMAND ----------
