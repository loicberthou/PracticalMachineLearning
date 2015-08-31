
### Q1

library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

library(caret)
library(doParallel)
library(randomForest)

# Create the cluster of workers
cl <- makeCluster(detectCores())
registerDoParallel(cl)

vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)

fit1 <- train(y ~ ., data = vowel.train, model = "rf")
fit2 <- train(y ~ ., data = vowel.train, model = "gbm")

stopCluster(cl)

predict1 <- predict(fit1, vowel.test)
predict2 <- predict(fit2, vowel.test)

confusionMatrix(predict1, vowel.test$y)$overall[1]
confusionMatrix(predict2, vowel.test$y)$overall[1]

predictCommon <- predict1[predict1 == predict2]
reference <- vowel.test$y[predict1 == predict2]

confusionMatrix(predictCommon, reference)$overall[1]

### Q2

library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

set.seed(62433)

fit3 <- train(diagnosis ~ ., data = training, model = "rf")
fit4 <- train(diagnosis ~ ., data = training, model = "gbm")
fit5 <- train(diagnosis ~ ., data = training, model = "lda")

predict3 <- predict(fit3, testing)
predict4 <- predict(fit4, testing)
predict5 <- predict(fit5, testing)

predDF <- data.frame(predict3, predict4, predict5, diagnosis = testing$diagnosis)
combModFit <- train(diagnosis ~ ., method = "rf", data = predDF)
combPred <- predict(combModFit, predDF)

confusionMatrix(predict3, testing$diagnosis)$overall[1]
confusionMatrix(predict4, testing$diagnosis)$overall[1]
confusionMatrix(predict5, testing$diagnosis)$overall[1]
confusionMatrix(combPred, testing$diagnosis)$overall[1]

### Q3

set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(233)

trainingMatrix <- as.matrix(training[, -9])
fit6 <- enet(x = trainingMatrix, y=CompressiveStrength)
plot(fit6, use.color=TRUE)

### Q4

library(lubridate)  # For year() function below
setwd("/Users/loicberthou/Documents/Dropbox/Programming/Coursera/Practical Machine Learning")
dat = read.csv("gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)

library(forecast)
#training$date <- as.Date(training$date, format="%Y-%m-%d")
#xtsTrain <- xts(training[,-c(1,2)], order.by=as.Date(training[,2], "%Y-%m-%d"))
#tsTrain <- as.ts(xtsTrain)
fit7 <- bats(tstrain)
fcast <- forecast(fit7, 235)

accDataFrame <- data.frame(fcast$lower[,2], fcast$upper[,2], testing$visitsTumblr)
accDataFrame$isAcc <- accDataFrame[,1] < accDataFrame[,3] & accDataFrame[,3] < accDataFrame[,2]
sum(accDataFrame$isAcc) / length(accDataFrame$isAcc)

### Q5

set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(325)

library(e1071)
fit8 <- svm(CompressiveStrength ~ ., data=training)
pred <- predict(fit8, testing)
accuracy(pred, testing$CompressiveStrength)[2]
