# Question #1
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)
library(rattle)
library(rpart)

data <- segmentationOriginal

training <- data[data$Case == 'Train',]
testing <- data[data$Case == 'Test',]

training$Case <- Null
testing$Case <- Null
training$Cell <- Null
testing$Cell <- Null

model <- train(Class ~ ., data = training, method = 'rpart')
fancyRpartPlot(model$finalModel)


# Question #3
library(pgmm)
library(caret)
library(tree)
data(olive)
olive = olive[,-1]

newdata = as.data.frame(t(colMeans(olive)))
model <- tree(Area ~ ., data = olive)

prediction <- predict(model, newdata = newdata)
prediction
summary(olive$Area)

# Question #4
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

set.seed(13234)
model <- train(chd ~ age + alcohol + obesity + tobacco + typea + ldl,
               data = trainSA,
               method="glm",
               family="binomial")

predictionTrain <- predict(model, trainSA)
predictionTest <- predict(model, testSA)

missClass = function(values,prediction){
        sum(((prediction > 0.5)*1) != values)/length(values)
}

missClassRateTrain <- missClass(trainSA$chd, predictionTrain)
missClassRateTest <- missClass(testSA$chd, predictionTest)

missClassRateTrain
missClassRateTest

# Question #5
library(ElemStatLearn)
library(caret)
data(vowel.train)
data(vowel.test) 

vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)

set.seed(33833)

model = train(y ~ .,
              data = vowel.train,
              method="rf")

imp <- varImp(model)
imp
