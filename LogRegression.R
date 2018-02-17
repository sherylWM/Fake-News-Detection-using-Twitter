TwitterLRData <- read.csv(file.choose(),header=T,stringsAsFactors = FALSE)
View(TwitterLRData)

library(aod)
library(ggplot2)

attach(TwitterLRData)
#detach(TwitterLRData)
sapply(TwitterLRData, class)

TwitterSubsetLRData <- subset(TwitterLRData, select=c(retweet_count, favorite_count,
                                                      length, user_statuses_count,
                                                      user_verified,statuses.followers_count,
                                                      friends.followers_count,user_has_url.,
                                                      no_of_question_marks,no_of_exclamation_marks,
                                                      no_of_hashtags, no_of_mentions,
                                                      no_of_urls,no_of_colon_marks,
                                                      no_of_words,polarity,no_of_firstOrderPronoun,
                                                      no_of_secondOrderPronoun, no_of_thirdOrderPronoun,
                                                      Source_Category,Age_of_UserAccount_indays,
                                                      Final.Label, Label))


TwitterSubsetLRData$Label <- as.factor(TwitterSubsetLRData$Label)
View(TwitterSubsetLRData)
samp_size <- floor(0.80 * nrow(TwitterSubsetLRData))
set.seed(2345)

#Converting character class to factors
TwitterSubsetLRData$user_has_url. <- as.factor(TwitterSubsetLRData$user_has_url.)
TwitterSubsetLRData$user_verified <- as.factor(TwitterSubsetLRData$user_verified)
TwitterSubsetLRData$Final.Label <- as.factor(TwitterSubsetLRData$Final.Label)
TwitterSubsetLRData$polarity <- as.factor(TwitterSubsetLRData$polarity)
TwitterSubsetLRData$Source_Category <- as.factor(TwitterSubsetLRData$Source_Category)

sapply(TwitterSubsetLRData, class)

tweet_ind <- sample(seq_len(nrow(TwitterSubsetLRData)), size = samp_size)
tweet_trainLR <- TwitterSubsetLRData[tweet_ind,]
tweet_testLR <- TwitterSubsetLRData[-tweet_ind,]

prop.table(table(tweet_trainLR$Final.Label))
prop.table(table(tweet_testLR$Final.Label))


logRegModel <- glm(Label~ retweet_count+
                     favorite_count+ length +
                     user_statuses_count
                     + user_verified + user_has_url.
                     + statuses.followers_count
                     + friends.followers_count
                     + no_of_question_marks
                     + no_of_exclamation_marks
                     + no_of_colon_marks
                     + no_of_hashtags + no_of_mentions
                     + no_of_urls + no_of_words + polarity + Source_Category
                     + no_of_firstOrderPronoun + no_of_secondOrderPronoun
                     + no_of_thirdOrderPronoun + Age_of_UserAccount_indays,
                     data = tweet_trainLR,
                     family = binomial)

logRegModel
summary(logRegModel)

View(tweet_trainLR)
View(tweet_testLR)
tweet_testLR$predict_label <- c(1:nrow(tweet_testLR))
tweet_testLR$predict_label <- NA
tweet_testLR$predict_label <- predict(logRegModel, newdata=tweet_testLR, type="response")

tweet_testLR$predict_final_label <- c(1:nrow(tweet_testLR))
tweet_testLR$predict_final_label <- ifelse(tweet_testLR$predict_label > 0.5, 1,0)

#calculating accuracy
#converting data frame to a matrix
m.tweet_testLR <- as.matrix(tweet_testLR)
View(m.tweet_testLR)
countLabels <- 0
for(i in 1:nrow(tweet_testLR))
{
  if(as.integer(m.tweet_testLR[i,23]) == as.integer(m.tweet_testLR[i,25]))
  {
    countLabels <- countLabels + 1
  }
}

#accuracy of our model
accuracy <- countLabels/nrow(tweet_testLR)
accuracy  #Gives 60% accuracy with Logistic Regression Model


library(gmodels)
CrossTable(tweet_testLR$Label ,tweet_testLR$predict_final_label, prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('Actual Labels', 'Predicted Labels'))


#******************************************************************************
#Model 2
# Using 10 fold cross validation
TwitterLRCVData <- subset(TwitterLRData, select=c(retweet_count, favorite_count,
                                                      length, user_statuses_count,
                                                      user_verified,statuses.followers_count,
                                                      friends.followers_count,user_has_url.,
                                                      no_of_question_marks,no_of_exclamation_marks,
                                                      no_of_hashtags, no_of_mentions,
                                                      no_of_urls,no_of_colon_marks,
                                                      no_of_words,polarity,no_of_firstOrderPronoun,
                                                      no_of_secondOrderPronoun, no_of_thirdOrderPronoun,
                                                      Source_Category,Age_of_UserAccount_indays,
                                                      Label))
k = 10 #no of folds
View(TwitterLRCVData)
# sample from 1 to k, nrow times (the number of observations in the data)
TwitterLRCVData$id <- sample(1:k, nrow(TwitterLRCVData), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

predictionTweetLRData <- data.frame()
testsetTweetLRData <- data.frame()

library(plyr)
#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingLRset <- subset(TwitterLRCVData, id %in% list[-i])
  testingLRset <- subset(TwitterLRCVData, id %in% c(i))
  
  # run a random forest model
  LRmodel <- glm(Label~ retweet_count+
                   favorite_count+ length +
                   user_statuses_count
                 + user_verified + user_has_url.
                 + statuses.followers_count
                 + friends.followers_count
                 + no_of_question_marks
                 + no_of_exclamation_marks
                 + no_of_colon_marks
                 + no_of_hashtags + no_of_mentions
                 + no_of_urls + no_of_words + polarity + Source_Category
                 + no_of_firstOrderPronoun + no_of_secondOrderPronoun
                 + no_of_thirdOrderPronoun + Age_of_UserAccount_indays,
                 data = trainingLRset,
                 family = binomial)
  
  # remove response column 22 Label
  temp <- as.data.frame(predict(LRmodel, newdata=testingLRset[,-22], type="response"))
  
  # append this iteration's predictions to the end of the prediction data frame
  predictionTweetLRData <- rbind(predictionTweetLRData, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetTweetLRData <- rbind(testsetTweetLRData, as.data.frame(testingLRset[,22]))
  
  progress.bar$step()
}

View(testsetTweetLRData)
View(predictionTweetLRData)
predictionTweetLRData$predictedLabel <- c(1:nrow(predictionTweetLRData))
predictionTweetLRData$predictedLabel <- ifelse(predictionTweetLRData$`predict(LRmodel, newdata = testingLRset[, -22], type = "response")` > 0.5, 1,0)

LRResult <- cbind(predictionTweetLRData$predictedLabel, testsetTweetLRData[,1])
names(LRResult) <- c("Predicted", "Actual")
View(LRResult)

class(LRResult)


#calculating accuracy

countLabels <- 0
for(i in 1:nrow(LRResult))
{
  if(LRResult[i,1] == LRResult[i,2])
  {
    countLabels <- countLabels + 1
  }
}

#accuracy of our model
accuracy <- countLabels/nrow(LRResult)
accuracy  # 57.6% with 10 fold cross validation

LRResultDF <- as.data.frame(LRResult)
View(LRResultDF)
class(LRResultDF)

library(gmodels)
CrossTable(LRResultDF$V2 ,LRResultDF$V1, prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('Actual Labels', 'Predicted Labels'))
