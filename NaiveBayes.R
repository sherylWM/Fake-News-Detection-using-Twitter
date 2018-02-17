TwitterNBData <- read.csv(file.choose(),header=T,stringsAsFactors = FALSE)
View(TwitterNBData)

library(e1071)
sapply(TwitterNBData, class)

attach(TwitterNBData)
TwitterSubsetNBData <- subset(TwitterNBData, select=c(retweet_count, favorite_count,
                                                      length, user_statuses_count,
                                                      user_verified,statuses.followers_count,
                                                      friends.followers_count,user_has_url.,
                                                      no_of_question_marks,no_of_exclamation_marks,
                                                      no_of_hashtags, no_of_mentions,
                                                      no_of_urls,no_of_colon_marks,
                                                      no_of_words,polarity,no_of_firstOrderPronoun,
                                                      no_of_secondOrderPronoun, no_of_thirdOrderPronoun,
                                                      Source_Category,Age_of_UserAccount_indays,
                                                      Final.Label))

TwitterSubsetNBData$Final.Label <- as.factor(TwitterSubsetNBData$Final.Label)
TwitterSubsetNBData$user_has_url. <- as.factor(TwitterSubsetNBData$user_has_url.)
TwitterSubsetNBData$user_verified <- as.factor(TwitterSubsetNBData$user_verified)
TwitterSubsetNBData$polarity <- as.factor(TwitterSubsetNBData$polarity)
TwitterSubsetNBData$Source_Category <- as.factor(TwitterSubsetNBData$Source_Category)

samp_size <- floor(0.80 * nrow(TwitterSubsetNBData))
set.seed(2345)

sapply(TwitterSubsetNBData, class)

tweet_ind <- sample(seq_len(nrow(TwitterSubsetNBData)), size = samp_size)
tweet_trainNB <- TwitterSubsetNBData[tweet_ind,]
tweet_testNB <- TwitterSubsetNBData[-tweet_ind,]

tweets_NB_classifier <- naiveBayes(Final.Label~., data = tweet_trainNB)
tweets_NB_predict <- predict(tweets_NB_classifier,tweet_testNB)

library(gmodels)

CrossTable(tweets_NB_predict,tweet_testNB$Final.Label,prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted','actual'))
#Accuracy of this model is 62%
AccuracyNB <- 31/50
AccuracyNB

#***************************************************************************
#Model 2
#Applying 10 fold cross validation

TwitterNBCVData <- subset(TwitterNBData, select=c(retweet_count, favorite_count,
                                                      length, user_statuses_count,
                                                      user_verified,statuses.followers_count,
                                                      friends.followers_count,user_has_url.,
                                                      no_of_question_marks,no_of_exclamation_marks,
                                                      no_of_hashtags, no_of_mentions,
                                                      no_of_urls,no_of_colon_marks,
                                                      no_of_words,polarity,no_of_firstOrderPronoun,
                                                      no_of_secondOrderPronoun, no_of_thirdOrderPronoun,
                                                      Source_Category,Age_of_UserAccount_indays,
                                                      Final.Label))

TwitterNBCVData$Final.Label <- as.factor(TwitterNBCVData$Final.Label)
TwitterNBCVData$user_has_url. <- as.factor(TwitterNBCVData$user_has_url.)
TwitterNBCVData$user_verified <- as.factor(TwitterNBCVData$user_verified)
TwitterNBCVData$polarity <- as.factor(TwitterNBCVData$polarity)
TwitterNBCVData$Source_Category <- as.factor(TwitterNBCVData$Source_Category)

k = 10 #no of folds

# sample from 1 to k, nrow times (the number of observations in the data)
TwitterNBCVData$id <- sample(1:k, nrow(TwitterNBCVData), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

predictionTweetNBData <- data.frame()
testsetTweetNBData <- data.frame()

library(plyr)

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingNBset <- subset(TwitterNBCVData, id %in% list[-i])
  testingNBset <- subset(TwitterNBCVData, id %in% c(i))
  
  # run a Naive Bayes model
  NBmodel <- naiveBayes(Final.Label~., data = trainingNBset)
  
  # remove response column 22
  temp <- as.data.frame(predict(NBmodel, testingNBset[,-22]))
  # append this iteration's predictions to the end of the prediction data frame
  predictionTweetNBData <- rbind(predictionTweetNBData, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetTweetNBData <- rbind(testsetTweetNBData, as.data.frame(testingNBset[,22]))
  
  progress.bar$step()
}

View(testsetTweetNBData)
NBResult <- cbind(predictionTweetNBData, testsetTweetNBData[,1])
names(NBResult) <- c("Predicted", "Actual")
View(NBResult)

class(NBResult)


#calculating accuracy
#converting data frame to a matrix
m.NBResult <- as.matrix(NBResult)
View(m.NBResult)
countLabels <- 0
for(i in 1:nrow(NBResult))
{
  if(m.NBResult[i,1] == m.NBResult[i,2])
  {
    countLabels <- countLabels + 1
  }
}

#accuracy of our model is 60.8
accuracyNBCV <- countLabels/nrow(NBResult)
accuracyNBCV

library(gmodels)
CrossTable(NBResult$Actual, NBResult$Predicted, prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('Actual Labels', 'Predicted Labels'))

AccuracyNCV <- (30 + 122)/250
AccuracyNCV 
