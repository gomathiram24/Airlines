# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:09:12 2018

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import json
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


ca_vid = pd.read_csv("C:/Users/Documents/YouTubeVideoAnalysis/Datasets/CAvideos.csv")
gb_vid = pd.read_csv("C:/Users/Documents/YouTubeVideoAnalysis/Datasets/GBvideos.csv",  error_bad_lines= False )
us_vid = pd.read_csv("C:/Users/Documents/YouTubeVideoAnalysis/Datasets/USvideos.csv", error_bad_lines = False)
us_vid.head(3)

blank_category={}
with open("C:/Users/Amit/Documents/YouTubeVideoAnalysis/Datasets/US_category_id.json","r") as d:
    
    data = json.load(d)
    for category in data["items"]:
        blank_category[category["id"]]=category["snippet"]["title"]
        
blank_category

us_vid["trending_date"]=pd.to_datetime(us_vid["trending_date"],format="%y.%d.%m")
us_vid["publish_time"]=pd.to_datetime(us_vid["publish_time"])


us_vid["Trending_Year"]=us_vid["trending_date"].apply(lambda time:time.year)
us_vid["Trending_Month"]=us_vid["trending_date"].apply(lambda time:time.month)
us_vid["Trending_Day"]=us_vid["trending_date"].apply(lambda time:time.day)
us_vid["Trending_Day_of_Week"]=us_vid["trending_date"].apply(lambda time:time.dayofweek)
us_vid["publish_Year"]=us_vid["publish_time"].apply(lambda time:time.year)
us_vid["publish_Month"]=us_vid["publish_time"].apply(lambda time:time.month)
us_vid["publish_Day"]=us_vid["publish_time"].apply(lambda time:time.day)
us_vid["publish_Day_of_Week"]=us_vid["publish_time"].apply(lambda time:time.dayofweek)
us_vid["Publish_Hour"]=us_vid["publish_time"].apply(lambda time:time.hour)

day_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
us_vid["publish_Day_of_Week"]=us_vid["publish_Day_of_Week"].map(day_map)
us_vid["Trending_Day_of_Week"]=us_vid["Trending_Day_of_Week"].map(day_map)
us_vid.head(2)

us_vid.info()

cat_list=["views,likes,dislikes,comment_count".split(",")]
for column in cat_list:
    us_vid[column]=us_vid[column].astype(int)

list2=["category_id"] 
for column in list2:
    us_vid[column]=us_vid[column].astype(str)
us_vid["Category"]=us_vid["category_id"].map(blank_category)


plt.style.use('ggplot')
plt.figure(figsize=(8,8))
list3=us_vid.groupby("Publish_Hour").count()["Trending_Year"].plot.bar()
list3.set_xticklabels(list3.get_xticklabels(),rotation=30)
plt.title("Publish Time of Videos")
sb.set_context(font_scale=1)


us_vid.head(2)

list5=us_vid[us_vid["Publish_Hour"]==17].groupby(["Category","publish_Day"]).count()["video_id"].unstack()
plt.figure(figsize=(9,9))#You can Arrange The Size As Per Requirement
sb.heatmap(list5)
plt.title("Category v/s Date Published on 17 hours")

plt.style.use('ggplot')
plt.figure(figsize=(8,8))
us_vid.groupby("Category").count()["views"].plot.bar()
plt.title("Category Wise Uploads")


plt.style.use('ggplot')
sb.set(rc={"figure.figsize":(20,10)})
us_vid[us_vid["Category"]=="Entertainment"].groupby(["views","title"]).count()[4108:]["video_id"].reset_index("views").plot.bar()
plt.title("Top 10 videos in Entertainment Category")

list6=sb.jointplot(x="publish_Day",y="Trending_Day",data=us_vid,size=8,kind="reg")
plt.title("Filter Out The Trending & Non Trending Videos")


plt.style.use('ggplot')
plt.figure(figsize=(8,8))
list7=us_vid["video_id"].value_counts().plot()
list7.set_xticklabels(list7.get_xticklabels(),rotation=90)
plt.title("This Show The The Occurance of Video in term of Id")


plt.style.use('ggplot')
list8=us_vid.groupby(["publish_Month","publish_Day_of_Week"]).count()["video_id"].unstack()
plt.figure(figsize=(12,10))
sb.heatmap(list8,cmap='viridis')


list10=us_vid[["title","views"]].sort_values(by="views",ascending=True)
list10.drop_duplicates("title",keep="last",inplace=True)
list11=list10.sort_values(by="views",ascending=False)
list12=list11.head(10)
list12.set_index("title",inplace=True)


plt.style.use('ggplot')
sb.set(rc={"figure.figsize":(10,10)})
list12.plot.barh()
plt.title("Most Watched Video on YouTube")

list13=us_vid[us_vid["title"].str.match("YouTube Rewind")]

sb.factorplot(x="video_id",y="likes",hue="Trending_Day",data=list13,size=8,kind="point")
plt.title("Trending Days v/s Like")


sb.factorplot(x="Trending_Day",y="views",hue="publish_Day",data=list13,size=8,kind="point")
plt.title("Trending Days and Views Analysis with Respect To Publish Day")


