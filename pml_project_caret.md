# Coursera Practical Machine Learning Course Project
Bill Cary  
Sunday, February 22, 2015  

# Title - Weight Lifting Exercises Prediction
## Synopsis
The goal of this project is to predict the manner in which participants did the
exercise. This is the "classe" variable in the training set. Predictive models
may use any of the other variables. The final output is a report describing how
the model was constructed, how cross validation was used, the expected out of
sample error, and why various choices were made. The model will also be used to
predict 20 different test cases. 

Using a Random Forest model with each tree consisting of five randomly chosen
features, I was able to obtain an average accuracy of 98% on an out-of-sample
test set.  The model was built using 2 repetitions of 10-fold cross validation
against 80% of the original training data, and then validated against tested
against an out-of-sample validation set consisting of the remaining 20% of the
original training data.

The model allowed me to correctly predict the classification of 19 of the 20
examples submitted as part of this assignment.

## Model Building
This exercise is an example of a classification problem.  (As opposed to a
regression problem.)  Rather than predicting a continuous variable (regression),
we are predicting a categorical variable.  Specifically, we are attempting to
predict the result of a weight lifting exercise.  In this case, the result is
one of the following five classifications: exactly according to the
specification (Class A), throwing the elbows to the front (Class B), lifting the
dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D)
and throwing the hips to the front (Class E).

### Set defaults
Set knitr to echo code by default

```r
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set echo=TRUE by default
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(knitr)
opts_chunk$set(echo = TRUE)
```

### Prepare the environment

```r
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import required libraries
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
suppressMessages(library(ggplot2))       # General plotting functions
suppressMessages(library(caret))         # Machine learning
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```r
suppressMessages(library(doParallel))    # Parallel processing
suppressMessages(library(pROC))          # Model performance measurement
```

```
## Warning: package 'pROC' was built under R version 3.1.2
```

```r
suppressMessages(library(randomForest))  # Machine learning
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```r
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Register parallel backend
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cl <- makeCluster(4)  # set appropriately for server on which job will run
registerDoParallel(cl)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set paths for files and directory structure
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Get the local path
path <- getwd()

## Define other paths
path_code <- paste0(path, "/code/")
path_train <- paste0(path, '/data/pml-training.csv')
path_test <- paste0(path, '/data/pml-testing.csv')
path_results <- paste0(path, '/results/')
path_model <- paste0(path, '/rfModel.rds')
```

### Load the data
Read the raw datasets into R.  When reading the data into R, I chose to treat
all of the following strings as NA: '', 'NA', '#DIV/0!'.  I then replaced NA
values with zero (which I believe to be acceptable for this particular dataset),
and then trimmed off the first six columns of the data, as I did not intend to
use those columns for modeling.  I also ensure that the classe variable is
stored in R as a factor data type, to ensure that later modeling efforts are
conducted as classification exercises, rather than regression.


```r
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read the data into R
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train <- read.csv(path_train, stringsAsFactors=FALSE,
                  na.strings = c('', 'NA', '#DIV/0!'))

test <- read.csv(path_test, stringsAsFactors=FALSE,
                 na.strings = c('', 'NA', '#DIV/0!'))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Replace NA values with 0
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train[is.na(train)] <- 0
test[is.na(test)] <- 0

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Ensure classe variable is stored as Factor
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train$classe <- as.factor(train$classe)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Drop features that will not be used for prediction
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train <- train[, 7:160]
test <- test[, 7:160]
```

### Partition the data
For this exercise, I chose to use 80% of the training records for actual model
building and training, and the remaining 20% for validation of the model.

```r
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Holdout 20% of training data for prelim testing/RMSLE estimates
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(107)
inTrain <- createDataPartition(y = train$classe,
                               p = 0.80,
                               list = FALSE)

training <- train[inTrain,]
testing <- train[-inTrain,]
```

Below is a very brief exploration of the classe variable we are attempting to
predict.  As seen below, class A occurs most frequently, with approximately
28% of the training set falling into that class.

```r
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Brief exploration of classe data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 4464 3038 2738 2573 2886
```

```r
prop.table(table(training$classe))
```

```
## 
##      A      B      C      D      E 
## 0.2843 0.1935 0.1744 0.1639 0.1838
```

### Data Cleansing

Before modeling, I performed a number of data cleansing/transformation steps to
simplify the input data and improve performance of the model.  The steps are
described below:

1. Remove columns with near-zero variance, as they will have little or no
predictive value.  I used the caret nearZeroVar function to identify the
columns in the training set having little or no variance.  I then removed these
columns from the training, testing and validation data sets.  This step removed
100 columns from the data sets.
2. Apply Principle Components Analysis (PCA) to the training set, then apply
the same PCA to the testing and validation sets.  This reduced the size of the
data sets from 54 columns down to 27 columns.

### Build and Train the Model
I chose to utilize the Caret package to facilitate model building/training.
For this exercise, I have chosen to use _10-fold cross validation_.

I chose to model the data using a Random Forest algorithm (rf in R).  I utilized
the caret package to handle cross validation and to select the optimal
tuning parameters (number of randomly chosen features in each tree).  Based on
offline tuning using caret, a value of 5 was identified as a good choice for the
mtry parameter.

The model results (within sample) are shown below:


```r
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Establish training control and tuning parameters
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Set up training control parameters (10-fold repeated cross validation)
fitControl <- trainControl(## 10-fold CV
        method = "repeatedcv",
        number = 2,
        ## repeated ten times
        repeats = 1,
        classProbs = TRUE)

grid <- expand.grid(mtry = c(5))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Drop columns with near zero variance (add no useful information for model)
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nzv <- nearZeroVar(training)
training <- training[, -nzv]
testing <- testing[, -nzv]
test <- test[, -nzv]

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Preprocess the data (perform PCA)
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
preprocValues <- preProcess(training[, -length(training)],
                            method = c('pca', 'center', 'scale'))

trainTransformed <- predict(preprocValues, training[, -length(training)])
testTransformed <- predict(preprocValues, testing[, -length(testing)])
validateTransformed <- predict(preprocValues, test[, -length(test)])

trainTransformedClasse <- training[, length(training)]
testTransformedClasse <- testing[, length(testing)]

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train model
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(300)

model <- train(x = trainTransformed
               ,y = trainTransformedClasse
               ,data = trainTransformed
               ,method = 'rf'
               ,trControl = fitControl
               ,tuneGrid = grid
               ,verbose = FALSE
               ,metric = 'Accuracy')

## Print the Model and Model Summary        
print(model)
```

```
## Random Forest 
## 
## 15699 samples
##    27 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (2 fold, repeated 1 times) 
## 
## Summary of sample sizes: 7850, 7849 
## 
## Resampling results
## 
##   Accuracy  Kappa   Accuracy SD  Kappa SD
##   0.9572    0.9458  0.00613      0.007739
## 
## Tuning parameter 'mtry' was held constant at a value of 5
## 
```

```r
summary(model)
```

```
##                 Length Class      Mode     
## call                6  -none-     call     
## type                1  -none-     character
## predicted       15699  factor     numeric  
## err.rate         3000  -none-     numeric  
## confusion          30  -none-     numeric  
## votes           78495  matrix     numeric  
## oob.times       15699  -none-     numeric  
## classes             5  -none-     character
## importance         27  -none-     numeric  
## importanceSD        0  -none-     NULL     
## localImportance     0  -none-     NULL     
## proximity           0  -none-     NULL     
## ntree               1  -none-     numeric  
## mtry                1  -none-     numeric  
## forest             14  -none-     list     
## y               15699  factor     numeric  
## test                0  -none-     NULL     
## inbag               0  -none-     NULL     
## xNames             27  -none-     character
## problemType         1  -none-     character
## tuneValue           1  data.frame list     
## obsLevels           5  -none-     character
```

```r
confusionMatrix(model)
```

```
## Cross-Validated (2 fold, repeated 1 times) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.0  0.5  0.1  0.0  0.0
##          B  0.2 18.3  0.5  0.0  0.3
##          C  0.1  0.4 16.6  1.0  0.2
##          D  0.1  0.0  0.2 15.2  0.3
##          E  0.0  0.1  0.1  0.1 17.6
```

### Evaluating Model Accuracy
Out-of-sample predictions (more indicative of true model performance against
unseen data) are shown below in the form of a confusion matrix and a ROC curve
plot.  At a minimum, we would expect the AUC (area under the ROC curve) to
be greater than 0.50 (the AUC expected from random guessing).  For practical
purposes, we would hope for an AUC in excess of 0.80, a value that is generally
considered to be "good."  In addition, we also expect out-of-sample accuracy to
exceed 28% - the accuracy we could resonably expect to obtain if we simply
predicted the most frequent class (A) for each testing sample.


```r
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate predictions against 20% of data held out for testing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test.predictions <- predict(model, testTransformed)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Print Out-of-Sample confusion matrix
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
confusionMatrix(test.predictions, testTransformedClasse)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1108   16    0    0    0
##          B    3  731   10    0    1
##          C    1    9  667   21    1
##          D    2    1    4  622    6
##          E    2    2    3    0  713
## 
## Overall Statistics
##                                         
##                Accuracy : 0.979         
##                  95% CI : (0.974, 0.983)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.974         
##  Mcnemar's Test P-Value : 0.000197      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.993    0.963    0.975    0.967    0.989
## Specificity             0.994    0.996    0.990    0.996    0.998
## Pos Pred Value          0.986    0.981    0.954    0.980    0.990
## Neg Pred Value          0.997    0.991    0.995    0.994    0.998
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.282    0.186    0.170    0.159    0.182
## Detection Prevalence    0.287    0.190    0.178    0.162    0.184
## Balanced Accuracy       0.994    0.979    0.983    0.982    0.993
```

```r
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate AUC
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
auc <- auc(as.numeric(testTransformedClasse),
           as.numeric(test.predictions))

auc
```

```
## Area under the curve: 0.984
```

As can be seen above, the performance of the model on out-of-sample data is
quite good.  Accuracy is approximately 98% - far greater than the 28% that would
be expected from always choosing the most frequent class.  In addition, AUC is
far above both the 0.50 expected from random guessing and the 0.80 generally
considered to be "good."  This is also demonstrated by the results obtained
when the model was applied to the offficial test data and submitted for scoring;
the correct result was obtained for 19 of the 20 examples in the test set.

## Suggestions for Improvement/Further Work
This analysis was based on the use of a random forest model.  While performance
was good, it would be interesting to compare the performance of the random
forest with that of SVM, GBM and neural net models.  In addition, it would be
interesting to utilize a deep learning neural network model.  I did not attempt
all of these models simply because of the processing time required to train
the models.  (The random forest model in this exercise ran overnight for a
period of several hours.)

In addition to the application of additional algorithms, it would also be
interesting to look at the sequencing of the measurements to incorporate any
temporal (time series) effects that may be present.

## Citation
Data for this analysis was provided by the Human Activity Recognition (HAR) project. The HAR website is located at http://groupware.les.inf.puc-rio.br/har. Please refer to the following citation for additional information on the dataset:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#sbia_paper_section#ixzz3QjclRIIW
