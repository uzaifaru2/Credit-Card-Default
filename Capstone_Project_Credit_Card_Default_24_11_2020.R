

library(tidyverse)
library(broom)
library(caret)
library(ipred)
library(lattice)
library(ggplot2)
library(caret)
library(MASS)
library(car)
library(knitr)
library(mice)
library(lattice)
library(reshape2)
library(skimr)
library(ROSE)
library(rpart)
library(rpart.plot)
library(earth)

# Reading the data

data <- read.csv('https://raw.githubusercontent.com/uzaifaru2/Credit-Card-Default/main/credit_card_default_data.csv')


# Exploring the data

dim(data)

# We are having 30,000 records with around 25 columns - variables ( including a customer ID column)
# The defaults  have been encoded as 1 and 0 otherwise 


# All the variables including the customer ID seems to have been considered as integers



# Looking at the data to see if any Inconsistency 

table(data$EDUCATION)

# There are data with 5 and 6 defined for Education  which hasn't been defined in the data 
# data understanding document. We might have to group them under 4 which  is others

table(data$MARRIAGE)

# here also we have many under 0 , which has not been defined , we can put it under 3 Which is Others


#replace 0's with NAN, replace others too
data$EDUCATION[data$EDUCATION == 0] <- 4
data$EDUCATION[data$EDUCATION == 5] <- 4
data$EDUCATION[data$EDUCATION == 6] <- 4
data$MARRIAGE[data$MARRIAGE == 0] <- 3

# Checking on whether changes have been Made

table(data$EDUCATION)
table(data$MARRIAGE)





# Let us work out certain Derived variables 
# One can be as innovative as possible  in Deriving these variables and these variables 
# Can actually effect the performance of the model itself also.

# The Derived Variables that we can look at are
# The increase in the Billing amount in september as compared to August
# The Billing amount in a percentage of the Credit limit
# Payment to Bill AMNT Ration

data2 <-data %>% mutate( inc_billing_amnt = (BILL_AMT1- BILL_AMT2)/BILL_AMT2 , 
                         bill_amnt_Limit = BILL_AMT1/LIMIT_BAL,
                         pay_bill_amnt_ratio = PAY_AMT1/BILL_AMT2)


# Now Another Problem , what needs to be done for cases which are  Giving NA's or Inf/-inf

# One Option is to remove these variables another option might be to replace the NA's by 
# 0's  or the maximum values
# Infy can be handled by replacing the divisor ( if it was a 0 by the median value) but it will end 
# up changing a variable itself which  we should avoid.
# Depending on the prevalence of these values

# write.csv(data2, file = "data2.csv")

# We are going to replace the NA's by 0 and the Infinities by the maximum value in that column


max(data2$inc_billing_amnt[is.finite(data2$inc_billing_amnt)])

data2$inc_billing_amnt[data2$inc_billing_amnt == Inf] <- max(data2$inc_billing_amnt
                                                             [is.finite(data2$inc_billing_amnt)])
data2$inc_billing_amnt[data2$inc_billing_amnt == -Inf] <- min(data2$inc_billing_amnt
                                                              [is.finite(data2$inc_billing_amnt)])
data2$inc_billing_amnt[is.na(data2$inc_billing_amnt) ] <- 0

# managing the Infinity Values for the  pay_bill_amnt_ratio both the negative and positive Infinities


max(data2$pay_bill_amnt_ratio[is.finite(data2$pay_bill_amnt_ratio)])
min(data2$pay_bill_amnt_ratio[is.finite(data2$pay_bill_amnt_ratio)])

data2$pay_bill_amnt_ratio[data2$pay_bill_amnt_ratio == Inf] <- max(data2$pay_bill_amnt_ratio[is.finite(data2$pay_bill_amnt_ratio)])
data2$pay_bill_amnt_ratio[data2$pay_bill_amnt_ratio == -Inf] <- min(data2$pay_bill_amnt_ratio[is.finite(data2$pay_bill_amnt_ratio)])
data2$pay_bill_amnt_ratio[is.na(data2$pay_bill_amnt_ratio) ] <- 0




# Checking for NA's

sapply(data2, function(x) sum(is.na(x)))



# lets remove the ID columns from the data set
data2 <- data2[ , -1]



# We see that certain variables which  should be categorical are stored as integers or 
# numeric manner
# These should be converted into factors for further analysis
# The variables which  should be factors are " Sex", " Education " , " Marriage",  
# Using them as numeric would give a false indication of their relevance for the End 
# Many of the variables like the History of the payment status can also be used as numeric
# but as numeric they are not significant as conveyed by the correlation matrix and 
# hence we are using  # these are factors


names <- c(2:4)

data2[,names] <- lapply(data2[,names] , factor)
# str(data2)
#One-Hot Encoding
# Creating dummy variables is converting a categorical variable to as many binary variables as 
#mhere are categories.
dmy <- dummyVars(" ~ .", data = data2,fullRank = T) 

train_transformed <- data.frame(predict(dmy, newdata = data2)) 
# # See the structure of the new dataset
# str(train_transformed)

#Converting the dependent variable back to categorical 
train_transformed$default_payment<-as.factor(train_transformed$default_payment)

# checking the same and the other variables 
#str(train_transformed)

#Splitting training set into two parts based on outcome: 75% and 25% 
set.seed(7)
index <- createDataPartition(train_transformed$default_payment, p = 0.75, list=FALSE, times = 1) 
trainSet <- train_transformed[ index,]
testSet <- train_transformed[-index,]


# Preparing the First Model - Logistic Regression Model



log.model <- glm(default_payment ~., data = trainSet, family = binomial)

summary(log.model)

# Two of the derived variables etc

log.predictions <- predict(log.model, testSet, type="response")

log.predictions.rd <- ifelse(log.predictions > 0.5, 1, 0)
log.predictions.rd <- as.factor(log.predictions.rd)



con_matrix_log_0.5 <- confusionMatrix(log.predictions.rd,testSet$default_payment)

# With This Type of Mode The Sensitivity Seems to be High but the  Precision seems to be lower.
# Accuracy is around 80 % but that could be the case , but we could have got an accuracy of around 
# 75 % by predicting non Defaults
# Let us see how we can change things , by Changing the default probability from 0.5 to say 0.25

log.predictions.rd1 <- ifelse(log.predictions > 0.25, 1, 0)
log.predictions.rd1 <- as.factor(log.predictions.rd1)




con_matrix_log_0.25 <- confusionMatrix(log.predictions.rd1,testSet$default_payment)




# Extracting the Accuracy , Sensitivity and Specificity from the Confusion Matrix Created

log.accuracy.0.5 <- con_matrix_log_0.5$overall[["Accuracy"]]
log.accuracy.0.25 <- con_matrix_log_0.25$overall[["Accuracy"]]
log.specificity.0.5 <- con_matrix_log_0.5$byClass[["Specificity"]]
log.specificity.0.25 <- con_matrix_log_0.25$byClass[["Specificity"]]
log.sensitivity.0.5 <- con_matrix_log_0.5$byClass[["Sensitivity"]]
log.sensitivity.0.25 <- con_matrix_log_0.25$byClass[["Sensitivity"]]



# First Let us Look at the Proportions of the Default in the Training Set

table(trainSet$default_payment)
prop.table(table(trainSet$default_payment))

# Creating the Under Sample equally Distributed training Set


train_under_sample <- ovun.sample(default_payment ~ ., data = trainSet, 
                                  method = "under", N = 9954, seed = 1)$data
table(train_under_sample$default_payment)


# Now Lets  start Building  Tree Based Models and observe whether we are able to improve on 
# The Accuracy or the other measures like Specificity

tree_undersample <- rpart(default_payment ~ ., method = "class",
                          data =  train_under_sample, control = rpart.control(cp = 0.001))

# Now Let us Plot the Tree's

plot(tree_undersample, uniform = TRUE)

# Add labels to the decision tree
text(tree_undersample)


# it Can be seen properly in a normal R Studio Console


# Now Lets  Prune the tree for the Model tree_undersample

# Plotting the cross-validated error rate as a function of the complexity parameter
plotcp(tree_undersample)

# Using  printcp() to identify for which complexity parameter 
# the cross-validated error rate is minimized.

printcp(tree_undersample)

# Creating  an index for of the row with the minimum xerror
index0 <- which.min(tree_undersample$cptable[ , "xerror"])

# Creating tree_min
tree_min <- tree_undersample$cptable[index0, "CP"]

#  Prune the tree using tree_min
ptree_undersample <- prune(tree_undersample, cp = tree_min)


# Changing the Prior Probabilities

tree_prior <- rpart(default_payment ~ ., method = "class",parms = list(prior=c(0.6, 0.4)),
                    control = rpart.control(cp = 0.001), 
                    data = trainSet)

# Plot the decision tree
plot(tree_prior, uniform = TRUE)

# Add labels to the decision tree
text(tree_prior)

# As the Trees get more complex, they are not clearly visible( The Nodes)

# Pruning the Trees with Changed Prior Probabilities

# Plot the cross-validated error rate as a function of the complexity parameter
plotcp(tree_prior)

# Use printcp() to identify for which complexity parameter the 
# cross-validated error rate is minimized.
printcp(tree_prior)

# Create an index for of the row with the minimum xerror
index <- which.min(tree_prior$cptable[ , "xerror"])

# Create tree_min
tree_min <- tree_prior$cptable[index, "CP"]

#  Prune the tree using tree_min
ptree_prior <- prune(tree_prior, cp = tree_min)

# Using  prp() to plot 

prp(ptree_prior)

# Tree with Loss Matrix


tree_loss_matrix <- rpart(default_payment ~ ., method = "class",
                          parms = list(loss = matrix(c(0, 10, 1, 0), ncol=2)),
                          control = rpart.control(cp = 0.001), data =  trainSet)

# Plot the decision tree
plot(tree_loss_matrix, uniform = TRUE)

# Add labels to the decision tree
text(tree_loss_matrix)


# Pruning the tree with the loss matrix

set.seed(345)
tree_loss_matrix  <- rpart(default_payment ~ ., method = "class", data = trainSet,
                           parms = list(loss=matrix(c(0, 10, 1, 0), ncol = 2)),
                           control = rpart.control(cp = 0.001))

# Plot the cross-validated error rate as a function of the complexity parameter
plotcp(tree_loss_matrix)

printcp(tree_loss_matrix)

# Create an index for of the row with the minimum xerror
index2 <- which.min(tree_loss_matrix$cptable[ , "xerror"])

# Create tree_min
tree_min2 <- tree_loss_matrix$cptable[index2, "CP"]
# Prune the tree using cp = 0.0012788
ptree_loss_matrix <- prune(tree_loss_matrix, cp = tree_min2)

# Use prp() and argument extra = 1 to plot the pruned tree
prp(ptree_loss_matrix, extra = 1)



# Pruning the Tree for the case with Weights 
# This vector contains weights of 1 for the non-defaults in the training set, 
# and weights of 3 for defaults in the training sets. 
# By specifying higher weights for default, the model will assign higher 
# importance to classifying defaults correctly.

# Creating a Vector of weights

case_weights <- ifelse(trainSet$default_payment == 1, 3,1 )

head(case_weights)
str(case_weights)

head(trainSet$default_payment)
set.seed(345)
tree_weights <- rpart(default_payment ~ ., method = "class",
                      data = trainSet, weights = case_weights,
                      control = rpart.control(minsplit = 5, minbucket = 2, cp = 0.001))

# Plot the cross-validated error rate for a changing cp
plotcp(tree_weights)

# Create an index for of the row with the minimum xerror
index <- which.min(tree_weights$cp[ , "xerror"])

# Create tree_min
tree_min <- tree_weights$cp[index, "CP"]

#  Prune the tree using tree_min
ptree_weights <- prune(tree_weights, tree_min)

# Plot the pruned tree using the rpart.plot()-package
prp(ptree_weights, extra = 1)


# Now Lets Predict on the test set using the different models that we have developed and 
#Compute the Confusion Matrices for the respective models

# Make predictions for each of the pruned trees using the test set.
pred_undersample <- predict(ptree_undersample, newdata = testSet,  type = "class")
pred_prior <- predict(ptree_prior, newdata = testSet,  type = "class")
pred_loss_matrix <- predict(ptree_loss_matrix, newdata = testSet,  type = "class")
pred_weights <- predict(ptree_weights, newdata = testSet,  type = "class")

# Now Lets Create the Confusion Matrices of the Resulting Models and observe the Accuracies 
#and the Specificity/Sensitivity Obtained

pred_undersample <- as.factor(pred_undersample)
pred_prior <- as.factor(pred_prior)
pred_loss_matrix <- as.factor(pred_loss_matrix)
pred_weights <- as.factor(pred_weights)




con_matrix_undersample <- confusionMatrix(pred_undersample,testSet$default_payment)
con_matrix_prior <- confusionMatrix(pred_prior,testSet$default_payment)
con_matrix_loss_matrix <- confusionMatrix(pred_loss_matrix,testSet$default_payment)
con_matrix_weights<- confusionMatrix(pred_weights,testSet$default_payment)

# Extracting the Accuracy, Specificity and the Sensitivity

ptree_undersample.accuracy <- con_matrix_undersample$overall[["Accuracy"]]
ptree_undersample.specificity <- con_matrix_undersample$byClass[["Specificity"]]
ptree_undersample.sensitivity <- con_matrix_undersample$byClass[["Sensitivity"]]

ptree_prior.accuracy <- con_matrix_prior$overall[["Accuracy"]]
ptree_prior.specificity <- con_matrix_prior$byClass[["Specificity"]]
ptree_prior.sensitivity <- con_matrix_prior$byClass[["Sensitivity"]]

ptree_loss_matrix.accuracy <- con_matrix_loss_matrix$overall[["Accuracy"]]
ptree_loss_matrix.specificity <- con_matrix_loss_matrix$byClass[["Specificity"]]
ptree_loss_matrix.sensitivity <- con_matrix_loss_matrix$byClass[["Sensitivity"]]

ptree_weights.accuracy <- con_matrix_weights$overall[["Accuracy"]]
ptree_weights.specificity <- con_matrix_weights$byClass[["Specificity"]]
ptree_weights.sensitivity <- con_matrix_weights$byClass[["Sensitivity"]]

## All the Accuracies, Specificity and Sensitivity has been stored in 
# separate variables since I Intend to compile these and print out a summarized sheet

# Boosting Algorithms

# Multivariate Adaptive  Regression Splines and AdaBoost

# Creating the Training Set and the Test Set
# Here we will be scaling the Data to see the performance of the models with Scaled Data
# We Try and Build These models on the Entire set as well as the Undersampled data set 
# and Observe The accuracies and the Specificity/Sensitivity Achieved 

# Training the Complete Data set

trainSet1 <- trainSet %>% mutate(default_payment = 
                                   factor(ifelse(default_payment == "1", "Yes", "No")))
preProcess_range_model <- preProcess(trainSet1, method='range')
trainSet1 <- predict(preProcess_range_model, newdata = trainSet1)

train_under_sample1 <- train_under_sample %>% mutate(default_payment = 
                                                       factor(ifelse(default_payment == "1", "Yes", "No")))

preProcess_range_model1 <- preProcess(train_under_sample1, method = 'range')
train_under_sample1 <- predict(preProcess_range_model1, newdata = train_under_sample1)

# Preparing the test Set

testSet1 <- testSet %>% mutate(default_payment = 
                                 factor(ifelse(default_payment == "1", "Yes", "No")))

# Scaling the Test Set for the Full Model

testSetFull <- predict(preProcess_range_model, newdata = testSet1)
testSet_under_sample1 <- predict(preProcess_range_model1, newdata = testSet1)

# Developing a MARS Model- Multivariate Adaptive Regression Spline  to
# Predict the defaults on the Full data , in the later part
# we would be developing and Testing  it on the undersample Data
# We would be developing this model using both Tune Length and Tune Grid on the 
# Undersampled Training Set
# we would be Observing the Accuracy and the Sensitivity as we as the Specificity to 
# understand the performance of the model
# Define the training control


fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = TRUE,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
) 


# Step 1: Tune hyper parameters by setting tuneLength
set.seed(100)
model_mars1 <- train(default_payment ~ ., data=trainSet1 , method='earth',
                     tuneLength = 5, metric='ROC', trControl = fitControl)
# model_mars1

# Step 2: Predict on testData and Compute the confusion matrix

predictedMars1 <- predict(model_mars1, testSetFull)

con_matrix_Mars <- confusionMatrix(predictedMars1,testSetFull$default_payment)

Mars.accuracy <- con_matrix_Mars$overall[["Accuracy"]]
Mars.specificity <- con_matrix_Mars$byClass[["Specificity"]]
Mars.sensitivity <- con_matrix_Mars$byClass[["Sensitivity"]]

# Doing the predictions Using a Tune Grid

# Step 1: Define the tuneGrid
marsGrid <-  expand.grid(nprune = c(2, 4, 6, 8, 10), 
                         degree = c(1, 2, 3))

# Step 2: Tune hyper parameters by setting tuneGrid
set.seed(100)
model_mars2 = train(default_payment ~ ., data=trainSet1 , method='earth', 
                    metric='ROC', tuneGrid = marsGrid, trControl = fitControl)
# model_mars2



# Step 3: Predict on testData and Compute the confusion matrix
predictedMars2 <- predict(model_mars2, testSetFull)

con_Matrix_Mars_Tuned <- confusionMatrix(predictedMars2,testSetFull$default_payment)



Mars_Tuned.accuracy <- con_Matrix_Mars_Tuned$overall[["Accuracy"]]
Mars_Tuned.specificity <- con_Matrix_Mars_Tuned$byClass[["Specificity"]]
Mars_Tuned.sensitivity <- con_Matrix_Mars_Tuned$byClass[["Sensitivity"]]


# Developing the MARS Model on the UNdersampled Data

#  Developing a MARS Model- Multivariate Adaptive Regression Spline  to Predict the defaults 
# on the under sampled Data
# We would be developing this model using both Tune Length and Tune Grid on the 
# Undersampled Training Set
# we would be Observing the Accuracy and the Sensitivity as we as the Specificity to 
# understand the performance of the model
# Define the training control


fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = TRUE,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
) 


# Step 1: Tune hyper parameters by setting tuneLength
set.seed(100)
model_mars1_under_sample  <- train(default_payment ~ ., data=train_under_sample1 ,
                                   method='earth', tuneLength = 5, metric='ROC', trControl = fitControl)
# model_mars1_under_sample

# Step 2: Predict on testData and Compute the confusion matrix

predictedMars1_under_sample <- predict(model_mars1_under_sample, testSet_under_sample1)

con_Matrix_Mars_Under_sample <- confusionMatrix(predictedMars1_under_sample, 
                                                testSet_under_sample1$default_payment)





Mars_Under_Sample.accuracy <- con_Matrix_Mars_Under_sample$overall[["Accuracy"]]
Mars_Under_Sample.specificity <- con_Matrix_Mars_Under_sample$byClass[["Specificity"]]
Mars_Under_Sample.sensitivity <-con_Matrix_Mars_Under_sample$byClass[["Sensitivity"]]

# Doing the predictions Using a Tune Grid

# Step 1: Define the tuneGrid
marsGrid <-  expand.grid(nprune = c(2, 4, 6, 8, 10), 
                         degree = c(1, 2, 3))

# Step 2: Tune hyper parameters by setting tuneGrid
set.seed(100)
model_mars2_under_sample = train(default_payment ~ ., data=train_under_sample1 , 
                                 method='earth', metric='ROC', tuneGrid = marsGrid, trControl = fitControl)

# model_mars2_under_sample



# Step 3: Predict on testData and Compute the confusion matrix
predictedMars2_under_sample <- predict(model_mars2_under_sample, testSet_under_sample1)

con_Matrix_Mars_Tuned_Under_sample <-confusionMatrix(predictedMars2_under_sample, 
                                                     testSet_under_sample1$default_payment)


Mars_Tuned_Under_Sample.accuracy <- con_Matrix_Mars_Tuned_Under_sample$overall[["Accuracy"]]
Mars_Tuned_Under_Sample.specificity <- con_Matrix_Mars_Tuned_Under_sample$byClass[["Specificity"]]
Mars_Tuned_Under_Sample.sensitivity <-con_Matrix_Mars_Tuned_Under_sample$byClass[["Sensitivity"]]

# Creating the Adaboost Model on the Full data


set.seed(100)

# Train the model using adaboost
model_adaboost1 <- train(default_payment ~ ., data=trainSet1, method='adaboost',
                         tuneLength=2, trControl = fitControl)
# model_adaboost1

predicted_adaboost1 <- predict(model_adaboost1, testSetFull)
con_Matrix_adaboost <-confusionMatrix(predicted_adaboost1, testSetFull$default_payment)

Adaboost.accuracy <- con_Matrix_adaboost$overall[["Accuracy"]]
Adaboost.specificity <- con_Matrix_adaboost$byClass[["Specificity"]]
Adaboost.sensitivity <-con_Matrix_adaboost$byClass[["Sensitivity"]]




set.seed(100)

# Train the model using adaboost
model_adaboost1_under_sample = train(default_payment ~ ., data=train_under_sample1, 
                                     method='adaboost', tuneLength=2, trControl = fitControl)

# model_adaboost1_under_sample

predicted_adaboost1_under_sample <- predict(model_adaboost1_under_sample, testSet_under_sample1)
confusionMatrix(predicted_adaboost1_under_sample, testSet_under_sample1$default_payment)

con_Matrix_adaboost_under_sample <-confusionMatrix(predicted_adaboost1_under_sample, testSet_under_sample1$default_payment)

Adaboost_under_sample.accuracy <- con_Matrix_adaboost_under_sample$overall[["Accuracy"]]
Adaboost_under_sample.specificity <- con_Matrix_adaboost_under_sample$byClass[["Specificity"]]
Adaboost_under_sample.sensitivity <-con_Matrix_adaboost_under_sample$byClass[["Sensitivity"]]


# Compiling the Results of all the Models


The_Models <- c('Logistic_Regression_Cut_Off_0.5', 'Logistic_Regression_Cut_Off_0.25',
                'Pruned_Tree_Under_Sample','Pruned_Tree_Prior_Probability',
                'Pruned_Tree_Loss_Matrix', 'Pruned_Tree_Weights', 
                'MARS', 'MARS_Tuned',
                'MARs_Under_Sample', 'MARS_Under_Sample',
                'Adaboost','Adaboost_Under_Sample')

Accuracy_models <- c(log.accuracy.0.5 , log.accuracy.0.25 ,ptree_undersample.accuracy , 
                     ptree_prior.accuracy , ptree_loss_matrix.accuracy, ptree_weights.accuracy , Mars.accuracy ,
                     Mars_Tuned.accuracy, Mars_Under_Sample.accuracy, Mars_Tuned_Under_Sample.accuracy, 
                     Adaboost.accuracy,Adaboost_under_sample.accuracy )


Specificity_models <- c(log.specificity.0.5 , log.specificity.0.25 ,ptree_undersample.specificity , 
                        ptree_prior.specificity , ptree_loss_matrix.specificity, ptree_weights.specificity , Mars.specificity ,
                        Mars_Tuned.specificity, Mars_Under_Sample.specificity, Mars_Tuned_Under_Sample.specificity, 
                        Adaboost.specificity,Adaboost_under_sample.specificity )


Sensitivity_models <- c(log.sensitivity.0.5 , log.sensitivity.0.25 ,ptree_undersample.sensitivity , 
                        ptree_prior.sensitivity , ptree_loss_matrix.sensitivity, ptree_weights.sensitivity , Mars.sensitivity ,
                        Mars_Tuned.sensitivity, Mars_Under_Sample.sensitivity, Mars_Tuned_Under_Sample.sensitivity, 
                        Adaboost.sensitivity,Adaboost_under_sample.sensitivity )

Results <- data.frame(The_Models, Accuracy_models,Sensitivity_models ,Specificity_models)

Results





