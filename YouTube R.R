library(xlsx)
library(OneR)
library(rattle)
library(caret)
library(ISLR)
library(class)
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)

Videos <- data.frame(read.csv("C:\\Users\\Amit\\Documents\\YouTubeVideoAnalysis\\VideoFinalData.csv"))


summary(Videos)

fillNAwithMedian <- function(x){
  na_index <- which(is.na(x))        
  median_x <- median(x, na.rm=T)
  x[na_index] <- median_x
  return(x)
}


Videos[,9,10,] <- data.frame(lapply(Videos, fillNAwithMedian))

na_count <-sapply(Youtube_rem_na, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count

summ
#NA Remove
Youtube_rem_na <- Videos[!is.na(Videos$channel_score), ]
#Youtube_rem_na$in_link_count <- videos1$in_links_count[!is.na(videos1$in_links_count)]
#

#Viral Categorization
Youtube_rem_na$viral [ Youtube_rem_na$view_rate > median(Youtube_rem_na$view_rate) ] <- 1


summary(Youtube_rem_na)


#Youtube_rem_na[Youtube_rem_na$viral == NA] <- 0

Youtube_rem_na$viral[is.na(Youtube_rem_na$viral )] <- 0
#

summary(Youtube_rem_na$channel_score)
summary(Youtube_rem_na$dislikes)


#Step Binning
Bin_Likes <- cut(Youtube_rem_na$likes, breaks = c(0,3294,12233,36407,2729392), labels = c("Very Low","Low", "Medium", "High"))
which(is.na(Bin_Likes))

Bin_Likes <- cut(Youtube_rem_na$likes, quantile(Youtube_rem_na$likes, (0:4)/4), labels=c("Very Low","Low", "Medium", "High"), include.lowest=FALSE)
#which(is.na(Bin_Likes))
Bin_Likes[is.na(Bin_Likes)] <- "High"
#Bin_Likes <- t(Bin_Likes)
Bin_Likes.df <- as.data.frame(Bin_Likes)
#head(Bin_Likes)
#summary(Bin_Likes)
#which(is.na(Bin_Likes))
#Bin_Dislike <- cut(Youtube_rem_na$dislikes, levels = c(1,2,3,4,5,6,'Others'), label= c('Very Low',''))
Bin_Dislike <- as.numeric(as.character(Youtube_rem_na$dislikes))
Bin_Dislike <- cut(Bin_Dislike, breaks = c(0,134,424,1338,1674420), labels = c("Very Low","Low", "Medium", "High"))
Bin_Dislike[is.na(Bin_Dislike)] <- "High"
Bin_Dislike.df <- as.data.frame(Bin_Dislike)
Bin_Like_Dislike.df <- cbind(Bin_Likes.df, Bin_Dislike.df)
summary(Bin_Dislike)

Bin_Comment_Count <- cut(Youtube_rem_na$comment_count, breaks = c(0,414,1310,3921,1361980), labels = c("Very Low","Low", "Medium", "High"))
Bin_Comment_Count[is.na(Bin_Comment_Count)] <- "High"
Bin_Comment_Count.df <- as.data.frame(Bin_Comment_Count)


Bin_trending_day <- cut(Youtube_rem_na$trending_day, breaks = c(1,8,16,24,32), labels = c("1st week","2nd week", "3rd Week", "4th Week"))
Bin_trending_day[is.na(Bin_trending_day)] <- "3rd Week"
Bin_trending_day.df <- as.data.frame(Bin_trending_day)


Bin_publish_day <- cut(Youtube_rem_na$publish_day, breaks = c(1,9,16,23,31), labels = c("1st week","2nd week", "3rd Week", "4th Week"))
Bin_publish_day[is.na(Bin_publish_day)] <- "3rd Week"
Bin_publish_day.df <- as.data.frame(Bin_publish_day)


Bin_Publish_hour <- cut(Youtube_rem_na$publish_hour, breaks = c(0, 4, 8, 12, 16, 20, 24), labels = c("0-4","4-8","8-12","12-16","16-20","20-24"))
Bin_Publish_hour[is.na(Bin_Publish_hour)] <- "4-8"
Bin_Publish_hour.df <- as.data.frame(Bin_Publish_hour)


Bin_Count_Tags <- cut(Youtube_rem_na$count_tags, breaks = c(1, 9, 18, 28, 69), labels = c("Very Low","Low","Medium","High"))
Bin_Count_Tags[is.na(Bin_Count_Tags)] <- "Medium"
Bin_Count_Tags.df <- data.frame(Bin_Count_Tags)


Bin_Count_Title <- cut(Youtube_rem_na$count_title, breaks = c(0, 5, 7, 10, 21), labels = c("Very Low","Low","Medium","High"))
Bin_Count_Title[is.na(Bin_Count_Title)] <- "Medium"
Bin_Count_Title.df <- as.data.frame(Bin_Count_Title)

Bin_in_links_count <- cut(Youtube_rem_na$count_title, breaks = c(0, 2, 6, 10, 82), labels = c("Very Low","Low","Medium","High"))
Bin_in_links_count[is.na(Bin_in_links_count)] <- "Medium"
Bin_in_links_count.df <- as.data.frame(Bin_in_links_count)


Youtube_rem_na$category_id <- as.factor(Youtube_rem_na$category_id)
Youtube_rem_na$trending_year <- as.factor(Youtube_rem_na$trending_year)
Youtube_rem_na$trending_month <- as.factor(Youtube_rem_na$trending_month)
Youtube_rem_na$publish_year <- as.factor(Youtube_rem_na$publish_year)
Youtube_rem_na$publish_month <- as.factor(Youtube_rem_na$publish_month)
Youtube_rem_na$publish_week <- as.factor(Youtube_rem_na$publish_week)


str(Youtube_rem_na)

YouTube_Classification.df <- as.data.frame(Youtube_rem_na[,c(6,20,21,24,25,26,27,34)])

YouTube_video_class.df <- data.frame(Bin_Likes.df,Bin_Dislike.df, Bin_Comment_Count.df, Bin_trending_day.df, Bin_publish_day.df,Bin_Publish_hour.df,Bin_Count_Tags.df, Bin_Count_Title.df,  YouTube_Classification.df )
which(is.na(YouTube_video_class.df))


#str(YouTube_video_class.df)

#head(YouTube_video_class.df)


#YouTube_Classification.df <-data.frame(Bin_Likes,Bin_Dislike, Bin_Comment_Count, Bin_trending_day, Bin_publish_day,Bin_Publish_hour,Bin_Count_Tags, Bin_Count_Title, YouTube_Classification.df) 
#head(YouTube_Classification.df)


#YouTube_Classification.df <- YouTube_Classification.df[!is.na(YouTube_Classification.df), ]
#which(is.na(YouTube_Classification.df))
#summary(YouTube_Classification.df)

set.seed(111)
train.index <- sample(row.names(YouTube_video_class.df), 0.6*dim(YouTube_video_class.df)[1])
valid.index <- setdiff(row.names(YouTube_video_class.df), train.index)
train.df <- YouTube_video_class.df[train.index, ]
valid.df <- YouTube_video_class.df[valid.index, ]

# accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))
# = YouTube_video_class.df$viral 
#for(i in 1:14) {
#  knn.pred <- knn(train = train.df, test = valid.df,cl = train.df$viral, k=10)
#  accuracy.df[i, 2] <- confusionMatrix(knn.pred, valid.df[, 3])$overall[1]
#}  


par(mfrow=c(1,1))

cartpred.ct <- rpart(viral ~ ., data=train.df, method = "class", cp = 0.001, minsplit = 500) 
prp(cartpred.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)
cartpred.ct

train.pred <- predict(cartpred.ct, train.df, type = "class")
valid.pred <- predict(cartpred.ct, valid.df, type = "class")

par(mfrow = c(1,2))
confusionMatrix(train.pred, train.df$viral)
confusionMatrix(valid.pred, valid.df$viral)
plot(pROC::roc(train.pred, train.df$viral), main = "Training ROC Curve" )
plot(pROC::roc(valid.pred, valid.df$viral), main = "Validation ROC Curve")


rocs <- c(roc(train.pred, train.df$viral),roc(valid.pred, valid.df$viral))

plot(rocs)

summary(train.pred)
summary(cartpred.ct)

