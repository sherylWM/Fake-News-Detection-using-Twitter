TwitterRFData <- read.csv(file.choose(),header=T,stringsAsFactors = FALSE)
View(TwitterRFData)

sapply(TwitterRFData, class)
attach(TwitterRFData)

TwitterSubsetRFData <- subset(TwitterRFData, select=c(retweet_count, favorite_count,
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
View(TwitterSubsetRFData)
sapply(TwitterSubsetRFData, class)

TwitterSubsetRFData$Final.Label <- as.factor(TwitterSubsetRFData$Final.Label)
TwitterSubsetRFData$user_has_url. <- as.factor(TwitterSubsetRFData$user_has_url.)
TwitterSubsetRFData$user_verified <- as.factor(TwitterSubsetRFData$user_verified)
TwitterSubsetRFData$polarity <- as.factor(TwitterSubsetRFData$polarity)
TwitterSubsetRFData$Source_Category <- as.factor(TwitterSubsetRFData$Source_Category)

View(TwitterSubsetRFData)
k = 10 #no of folds

# sample from 1 to k, nrow times (the number of observations in the data)
TwitterSubsetRFData$id <- sample(1:k, nrow(TwitterSubsetRFData), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

predictionTweetData <- data.frame()
testsetTweetData <- data.frame()

library(plyr)
library(randomForest)
#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingRFset <- subset(TwitterSubsetRFData, id %in% list[-i])
  testingRFset <- subset(TwitterSubsetRFData, id %in% c(i))
  
  # run a random forest model
  RFmodel <- randomForest(trainingRFset$Final.Label ~ ., data = trainingRFset, ntree = 100)
  
  # remove response column 1, Sepal.Length
  temp <- as.data.frame(predict(RFmodel, testingRFset[,-22]))
  # append this iteration's predictions to the end of the prediction data frame
  predictionTweetData <- rbind(predictionTweetData, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetTweetData <- rbind(testsetTweetData, as.data.frame(testingRFset[,22]))
  
  progress.bar$step()
}

View(testsetTweetData)
RFResult <- cbind(predictionTweetData, testsetTweetData[,1])
names(RFResult) <- c("Predicted", "Actual")
View(RFResult)

class(RFResult)


#calculating accuracy
#converting data frame to a matrix
m.RFResult <- as.matrix(RFResult)
View(m.RFResult)
countLabels <- 0
for(i in 1:nrow(RFResult))
{
  if(m.RFResult[i,1] == m.RFResult[i,2])
  {
    countLabels <- countLabels + 1
  }
}

#accuracy of our model
accuracy <- countLabels/nrow(RFResult)
accuracy

library(gmodels)
CrossTable(RFResult$Actual, RFResult$Predicted, prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('Actual Labels', 'Predicted Labels'))

AccuracyCT <- (34 + 130)/250
AccuracyCT #Gives 64.8% accuracy with 10 fold CV for RF model
