TwitterSVMData <- read.csv(file.choose(),header=T,stringsAsFactors = FALSE)
View(TwitterSVMData)

library(kernlab)
sapply(TwitterSVMData, class)

attach(TwitterSVMData)
TwitterSubsetSVMData <- subset(TwitterSVMData, select=c(retweet_count, favorite_count,
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

TwitterSubsetSVMData$Final.Label <- as.factor(TwitterSubsetSVMData$Final.Label)
samp_size <- floor(0.80 * nrow(TwitterSubsetSVMData))
set.seed(2345)

TwitterSubsetSVMData$user_has_url. <- as.factor(TwitterSubsetSVMData$user_has_url.)
TwitterSubsetSVMData$user_verified <- as.factor(TwitterSubsetSVMData$user_verified)
TwitterSubsetSVMData$Final.Label <- as.factor(TwitterSubsetSVMData$Final.Label)
TwitterSubsetSVMData$polarity <- as.factor(TwitterSubsetSVMData$polarity)
TwitterSubsetSVMData$Source_Category <- as.factor(TwitterSubsetSVMData$Source_Category)

sapply(TwitterSubsetSVMData, class)


tweet_ind <- sample(seq_len(nrow(TwitterSubsetSVMData)), size = samp_size)
tweet_trainSVM <- TwitterSubsetSVMData[tweet_ind,]
tweet_testSVM <- TwitterSubsetSVMData[-tweet_ind,]

tweet_svm_classifier <- ksvm(Final.Label~., data = tweet_trainSVM, kernel = "vanilladot")
tweet_svm_classifier

tweet_svm_prediction <- predict(tweet_svm_classifier, tweet_testSVM)

tweet_svm_results <- table(tweet_svm_prediction, tweet_testSVM$Final.Label)

library(caret)
confusionMatrix(tweet_svm_results)

#SVM classifier with vanilladot kernel gives 70% accuracy

#******************************************************************************
#Model 2 with Radial Basis Function
tweet_svm_classifier2 <- ksvm(Final.Label~., data = tweet_trainSVM, kernel = "rbfdot")
tweet_svm_classifier2

tweet_svm_prediction2 <- predict(tweet_svm_classifier2, tweet_testSVM)

tweet_svm_results2 <- table(tweet_svm_prediction2, tweet_testSVM$Final.Label)

library(caret)
confusionMatrix(tweet_svm_results2)

#SVM classifier with radial basis kernel function gives 62% accuracy

#*******************************************************************************
#Model 3 with 10 fold CV

TwitterSVMCVData <- subset(TwitterSVMData, select=c(retweet_count, favorite_count,
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

sapply(TwitterSVMCVData, class)

TwitterSVMCVData$Final.Label <- as.factor(TwitterSVMCVData$Final.Label)
TwitterSVMCVData$user_has_url. <- as.factor(TwitterSVMCVData$user_has_url.)
TwitterSVMCVData$user_verified <- as.factor(TwitterSVMCVData$user_verified)
TwitterSVMCVData$polarity <- as.factor(TwitterSVMCVData$polarity)
TwitterSVMCVData$Source_Category <- as.factor(TwitterSVMCVData$Source_Category)

View(TwitterSVMCVData)
k = 10 #no of folds

# sample from 1 to k, nrow times (the number of observations in the data)
TwitterSVMCVData$id <- sample(1:k, nrow(TwitterSVMCVData), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

predictionTweetSVMData <- data.frame()
testsetTweetSVMData <- data.frame()

library(plyr)

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingSVMset <- subset(TwitterSVMCVData, id %in% list[-i])
  testingSVMset <- subset(TwitterSVMCVData, id %in% c(i))
  
  # run a SVM classifier
  SVMmodel <- ksvm(Final.Label~., data = trainingSVMset, kernel = "rbfdot")
  
  # remove response column 22
  temp <- as.data.frame(predict(SVMmodel, testingSVMset[,-22]))
  # append this iteration's predictions to the end of the prediction data frame
  predictionTweetSVMData <- rbind(predictionTweetSVMData, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetTweetSVMData <- rbind(testsetTweetSVMData, as.data.frame(testingSVMset[,22]))
  
  progress.bar$step()
}

View(testsetTweetSVMData)
SVMResult <- cbind(predictionTweetSVMData, testsetTweetSVMData[,1])
names(SVMResult) <- c("Predicted", "Actual")
View(SVMResult)

class(SVMResult)


#calculating accuracy
#converting data frame to a matrix
m.SVMResult <- as.matrix(SVMResult)
View(m.SVMResult)
countLabels <- 0
for(i in 1:nrow(SVMResult))
{
  if(m.SVMResult[i,1] == m.SVMResult[i,2])
  {
    countLabels <- countLabels + 1
  }
}

#accuracy of SVM model
#accuracy <- countLabels/nrow(SVMResult)
#accuracy

library(gmodels)
CrossTable(SVMResult$Actual, SVMResult$Predicted, prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('Actual Labels', 'Predicted Labels'))

AccuracySVMCV <- (7 + 144)/250
AccuracySVMCV #Gives 60.4% accuracy with 10 fold CV for RF model

