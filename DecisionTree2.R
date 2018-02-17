TwitterDTData <- read.csv(file.choose(),header= T,stringsAsFactors = FALSE)
View(TwitterDTData)
attach(TwitterDTData)

TwitterSubsetDTData <- subset(TwitterDTData, select=c(retweet_count, favorite_count,
                                                      length, user_statuses_count,
                                                      user_verified,statuses.followers_count,
                                                      friends.followers_count,user_has_url.,
                                                      no_of_question_marks,no_of_exclamation_marks,
                                                      no_of_hashtags, no_of_mentions,
                                                      no_of_urls,no_of_colon_marks,
                                                      no_of_words,polarity,no_of_firstOrderPronoun,
                                                      no_of_secondOrderPronoun, no_of_thirdOrderPronoun,
                                                      Source_Category,Age_of_UserAccount_indays,
                                                      Final.Label)
                              )

View(TwitterSubsetDTData)
library(caret)
library(mlbench)
samp_size <- floor(0.80 * nrow(TwitterSubsetDTData))
set.seed(1235)

table(TwitterSubsetDTData$Final.Label)

tweet_ind <- sample(seq_len(nrow(TwitterSubsetDTData)), size = samp_size)

#splitting into 80:20
tweet_train <- TwitterSubsetDTData[tweet_ind,]
tweet_test <- TwitterSubsetDTData[-tweet_ind,]

prop.table(table(tweet_train$Final.Label))
prop.table(table(tweet_test$Final.Label))

library(C50) 

#determining class of all columns in tweetTraining dataframe
sapply(tweet_train,class)

#converting logical value to factor
tweet_train$user_verified <- as.factor(tweet_train$user_verified)
tweet_train$user_has_url. <- as.factor(tweet_train$user_has_url.)
tweet_train$polarity <- as.factor(tweet_train$polarity)
tweet_train$Source_Category <- as.factor(tweet_train$Source_Category)
tweet_train$Final.Label <- as.factor(tweet_train$Final.Label)

View(tweet_train)

decisionTreeModel <- C50::C5.0(tweet_train[,c(1:21)],tweet_train$Final.Label)
decisionTreeModel
summary(decisionTreeModel) 

#Accuracy of our training Decision Tree model is 68.4


plot(decisionTreeModel)

#converting logical to factor for testing data set
tweet_test$user_verified <- as.factor(tweet_test$user_verified)
tweet_test$user_has_url. <- as.factor(tweet_test$user_has_url.)
tweet_test$polarity <- as.factor(tweet_test$polarity)
tweet_test$Source_Category <- as.factor(tweet_test$Source_Category)
tweet_test$Final.Label <- as.factor(tweet_test$Final.Label)

tweet_pred <- predict(decisionTreeModel,tweet_test)
library(gmodels)

#cross tabulation of predicted versus actual classes
CrossTable(tweet_test$Final.Label,tweet_pred,prop.chisq=FALSE, prop.c=FALSE,
           prop.r=FALSE, dnn=c('actual label', 'predicted label'))

#Accuracy of our Decision Tree model on testing dataset is 54%

#******************************************************************************
#Model 2
#Applying boosting

decisionTreeModelBoost <- C50::C5.0(tweet_train[,c(1:21)],tweet_train$Final.Label,trials=10)
decisionTreeModelBoost
summary(decisionTreeModelBoost) 

tweet_pred_boost <- predict(decisionTreeModelBoost,tweet_test)
library(gmodels)

#cross tabulation of predicted versus actual classes
CrossTable(tweet_test$Final.Label,tweet_pred_boost,prop.chisq=FALSE, prop.c=FALSE,
           prop.r=FALSE, dnn=c('Actual labels', 'Predicted labels'))

#Accuracy of our Decision Tree model after boosting on testing dataset is 60%

#******************************************************************************
#Model 3
#Applying 10 fold cross validation
TwitterDTCVData <- subset(TwitterDTData, select=c(retweet_count, favorite_count,
                                                      length, user_statuses_count,
                                                      user_verified,statuses.followers_count,
                                                      friends.followers_count,user_has_url.,
                                                      no_of_question_marks,no_of_exclamation_marks,
                                                      no_of_hashtags, no_of_mentions,
                                                      no_of_urls,no_of_colon_marks,
                                                      no_of_words,polarity,no_of_firstOrderPronoun,
                                                      no_of_secondOrderPronoun, no_of_thirdOrderPronoun,
                                                      Source_Category,Age_of_UserAccount_indays,
                                                      Final.Label)
)

TwitterDTCVData$Final.Label <- as.factor(TwitterDTCVData$Final.Label)
TwitterDTCVData$user_has_url. <- as.factor(TwitterDTCVData$user_has_url.)
TwitterDTCVData$user_verified <- as.factor(TwitterDTCVData$user_verified)
TwitterDTCVData$polarity <- as.factor(TwitterDTCVData$polarity)
TwitterDTCVData$Source_Category <- as.factor(TwitterDTCVData$Source_Category)

View(TwitterDTCVData)
k = 10 #no of folds

# sample from 1 to k, nrow times (the number of observations in the data)
TwitterDTCVData$id <- sample(1:k, nrow(TwitterDTCVData), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

predictionTweetDTData <- data.frame()
testsetTweetDTData <- data.frame()

library(plyr)

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingDTset <- subset(TwitterDTCVData, id %in% list[-i])
  testingDTset <- subset(TwitterDTCVData, id %in% c(i))
  
  # run a random forest model
  DTmodel <- C50::C5.0(trainingDTset[,c(1:21)],trainingDTset$Final.Label)
  
  # remove response column Final Label which is column no 22
  temp <- as.data.frame(predict(DTmodel, testingDTset[,-22]))
  # append this iteration's predictions to the end of the prediction data frame
  predictionTweetDTData <- rbind(predictionTweetDTData, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetTweetDTData <- rbind(testsetTweetDTData, as.data.frame(testingDTset[,22]))
  
  progress.bar$step()
}



View(testsetTweetDTData)
DTResult <- cbind(predictionTweetDTData, testsetTweetDTData[,1])
names(DTResult) <- c("Predicted", "Actual")
View(DTResult)

class(DTResult)


#calculating accuracy
#converting data frame to a matrix
m.DTResult <- as.matrix(DTResult)
View(m.DTResult)
countLabels <- 0
for(i in 1:nrow(DTResult))
{
  if(m.DTResult[i,1] == m.DTResult[i,2])
  {
    countLabels <- countLabels + 1
  }
}

#accuracy of our model is 62.4% after using 10 fold cross validation
accuracy <- countLabels/nrow(DTResult)
accuracy

library(gmodels)
CrossTable(DTResult$Actual, DTResult$Predicted, prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('Actual Labels', 'Predicted Labels'))

#Another way to calculate accuracy
AccuracyDT <- (34 + 122)/250
AccuracyDT #Gives 64.8% accuracy with 10 fold CV for RF model

