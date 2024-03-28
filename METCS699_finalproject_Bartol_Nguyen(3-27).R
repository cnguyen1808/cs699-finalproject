###############################################################################
# MET CS699
# Final Project
# Brandon Bartol & Chuong Nguyen
###############################################################################
# install.packages("GGally")
# install.packages("modeest")
# install.packages("dplyr")
# install.packages("caret")
# install.packages("Boruta")
# install.packages("ROSE")
library(rsample)
library(GGally)
library(modeest)
library(dplyr)
library(caret)
library(Boruta)
library(ROSE)
library(caret) 
library(e1071)
library(pROC)

#dir <- file.path("C:", "Users", "Admin", "iCloudDrive",
                 "CS699 - Data Mining", "FinalProject", "git")
setwd(dir)

 setwd("C:\\Users\\brand\\OneDrive\\Documents\\699\\TermProject\\Data")

Project_Data <- read.csv("project_dataset_5K.csv",header = TRUE)
head(Project_Data)

Project_Data_Cleansed <- Project_Data
sapply(Project_Data_Cleansed, class)

###############################################################################
# Part I: Data preprocessing
###############################################################################

#Count missing data in each column
sapply(Project_Data_Cleansed, function(x) sum(is.na(x)))
#remove columns with more than 90% of the data present
Project_Data_Cleansed2 <- Project_Data_Cleansed %>%
  select_if(function(x) sum(is.na(x)) < 1000) # INCREASED from 500 to keep 2 columns with 508 na values


sapply(Project_Data_Cleansed2, function(x) sum(is.na(x)))


#Add data to columns with null values:
#if column contains <= 5 unique values, then the mode is applied
#else the median is applied
#this is stratified for the Class column
grouped <- split(Project_Data_Cleansed2, Project_Data_Cleansed2$Class)

for (col_name in colnames(Project_Data_Cleansed2)) {
  unique_values <- unique(Project_Data_Cleansed2[[col_name]])
  
  if (length(unique_values) <= 5) {
    # Compute mode value for each group
    mode_value <- sapply(grouped, function(group) {
      mfv(group[[col_name]][!is.na(group[[col_name]])])
    })
    Project_Data_Cleansed2[[col_name]] <- ifelse(is.na(Project_Data_Cleansed2[[col_name]]), mode_value[Project_Data_Cleansed2$Class], Project_Data_Cleansed2[[col_name]])
  } else {
    # Compute median value for each group
    non_missing_values <- Project_Data_Cleansed2[[col_name]][!is.na(Project_Data_Cleansed2[[col_name]])]
    median_value <- median(non_missing_values)
    Project_Data_Cleansed2[[col_name]] <- ifelse(is.na(Project_Data_Cleansed2[[col_name]]), median_value, Project_Data_Cleansed2[[col_name]])
  }
}


#removing columns with near zero variance, changed frequency to 20 to match class notes
nearZeroVar(Project_Data_Cleansed2, freqCut = 20, saveMetrics =  TRUE)
near_zero_vars <- nearZeroVar(Project_Data_Cleansed2, freqCut = 20, saveMetrics = FALSE)
Project_Data_Cleansed2 <- Project_Data_Cleansed2[, -near_zero_vars]

#Scale the data

scale(Project_Data_Cleansed2[,-ncol(Project_Data_Cleansed2)])
Project_Data_Cleansed2[,-ncol(Project_Data_Cleansed2)] <- scale(Project_Data_Cleansed2[,-ncol(Project_Data_Cleansed2)])

sapply(Project_Data_Cleansed2, class)
head(Project_Data_Cleansed2)


###############################################################################
# Part II: Apply feature selection and unbalanced-data treatment models
###############################################################################

# Split processed data set into training and testing data sets
# This test data set will be used in the very final evaluation step
splitting.df <- Project_Data_Cleansed2
splitting.df$Class <- factor(splitting.df$Class, levels = c("N", "Y"))

set.seed(31)
outer.split <- initial_split(splitting.df, prop = 0.8, strata = Class)
main.train <- training(outer.split)
main.test <- testing(outer.split)

Project_Data_Cleansed2 <- main.train


###########################
# Feature selection 1 -Brandon
###########################

#find highly correlated columns to remove duplicate influence (corr > .7)
corr <- cor(Project_Data_Cleansed2[,-ncol(Project_Data_Cleansed2)])
findCorrelation(corr, cutoff = 0.7, names = TRUE)
CorrColumnRemoved <- findCorrelation(corr, cutoff = 0.7, names = FALSE)
Project_Data_Cleansed2 <- Project_Data_Cleansed2[, -CorrColumnRemoved]
head(Project_Data_Cleansed2)




###########################
# Feature selection 2
###########################

#Use boruta to identify unimportant attributes
Project_Data_Cleansed3 <- Project_Data_Cleansed2
Project_Data_Cleansed3$Class <- factor(Project_Data_Cleansed2$Class, levels = c("N", "Y"))
borutaResult <- Boruta(Class ~ ., data = Project_Data_Cleansed3)
borutaResult$finalDecision

#remove rejected columns
BorutaRejectedColumns <- names(borutaResult$finalDecision[borutaResult$finalDecision == "Rejected"])
Project_Data_Cleansed3 <- Project_Data_Cleansed3[,!(names(Project_Data_Cleansed3) %in% BorutaRejectedColumns)]
colnames(Project_Data_Cleansed3)


###########################
# Feature selection 3
###########################

#Used Code provided in Class to do PCA
#df <- Project_Data_Cleansed3[, -(1:9)] - Why was 1:9? - Brandon
df <- test.overunder.df


head(df)
sapply(df, class)



# build Weka’s J48 decision tree model using all attributes
library(RWeka)
J48.model <- J48(Class ~ . , data=df)
# test the model on the test dataset
#pred <- predict(J48.model, newdata = test, type = "class")
#performance_measures  <- confusionMatrix(data=pred,
#                                         reference = test$Class)
#performance_measures

# apply PCA on the training dataset
pc <- prcomp(df[, -ncol(df)] %>% 
               select_if(~ all(is.numeric(.))), 
             center = TRUE, scale = TRUE) # exclude class attribute
summary(pc)
head(pc)

test_scores <- predict(pc, newdata = main.test)
test_scores
#PCA finds 32 principal components, we can preserve 90% of the total variability.

# first map (project) original attributes to new attributes created by PCA
tr <- predict(pc, training)
tr <- data.frame(tr, training[,ncol(training)])
ts <- predict(pc, test)
ts <- data.frame(ts, test[,ncol(test)])
colnames(tr)[ncol(tr)] <- 'class'
colnames(ts)[ncol(ts)] <- 'class'
head(tr)

# Build model using only the first 32 components, test, get confusion matrix and accuracy
J48.model <- J48(class~., data=tr[c(1:32, ncol(tr))])
pred <- predict(J48.model, newdata = ts, type = "class")
performance_measures  <- confusionMatrix(data=pred,
                                         reference = ts$class)
performance_measures


###############################################
# Balance with Oversampling and Undersampling - Chuong
###############################################

balancing.df <- Project_Data_Cleansed3
balancing.df$Class <- factor(balancing.df$Class)
sapply(balancing.df, class)


###################
# (1) over-sampling 
###################
over_balance <- function(imbalanced.df, target_col) {
  
  imbalanced.df <<- imbalanced.df
  
  # Perform over-sampling + under-sampling
  sampled_df <- ROSE::ovun.sample(Class ~ ., 
                                  data = as.data.frame(imbalanced.df),
                                  method = "over", 
                                  p = 0.5,
                                  seed = 31)$data
  
  return(sampled_df)
}


test.over.df <- over_balance(as.data.frame(balancing.df))
table(test.over.df$Class)


####################
# (2) under-sampling 
####################
under_balance <- function(imbalanced.df, target_col) {
  
  imbalanced.df <<- imbalanced.df
  
  # Perform over-sampling + under-sampling
  sampled_df <- ROSE::ovun.sample(Class ~ ., 
                                  data = as.data.frame(imbalanced.df),
                                  method = "under", 
                                  p = 0.5,
                                  seed = 31)$data
  
  return(sampled_df)
}


test.under.df <- under_balance(as.data.frame(balancing.df))
table(test.under.df$Class)

###########################
# (3) over + under-sampling 
###########################
combine_sampling_balance <- function(imbalanced.df, target_col) {
  
  imbalanced.df <<- imbalanced.df
  
  # Perform over-sampling + under-sampling
  sampled_df <- ROSE::ovun.sample(Class ~ ., 
                                  data = as.data.frame(imbalanced.df),
                                  method = "both", 
                                  p = 0.5,
                                  seed = 31,
                                  N = nrow(imbalanced.df))$data
  
  return(sampled_df)
}


test.overunder.df <- combine_sampling_balance(as.data.frame(balancing.df))
table(test.overunder.df$Class)


##################################
# (4) Bootstrap balancing - Chuong
##################################
# Function to perform bootstrap resampling to balance dataset
bootstrap_balance <- function(df, target_col, balance_ratio) {
  # Split dataset into majority and minority classes
  majority_class <- df[df[[target_col]] == "N", ]
  minority_class <- df[df[[target_col]] == "Y", ]
  
  # Calculate the number of samples needed from the minority class
  minority_count <- nrow(minority_class)
  majority_count <- nrow(majority_class)
  target_minority_count <- balance_ratio * majority_count
  
  # Generate bootstrap samples from the minority class
  balanced_data <- minority_class[sample(1:minority_count, 
                                         target_minority_count, 
                                         replace = TRUE), ]
  
  # Combine with majority class
  balanced_data <- rbind(balanced_data, majority_class)
  
  return(balanced_data)
}

test.bootstrap.df <- bootstrap_balance(main.train, "Class", 1)
table(test.bootstrap.df$Class)

##############################################
# Data visualizations -Chuong
##############################################
# Pairwise


# Assuming 'dataset' is your dataframe



###############################################################################
# Part III: Model Building
###############################################################################

# Define a function to compute metrics and build the result table
# The factor in Boruta is currently give No as Positive Class
# While the goal of the project is identifying people with disorder "Y"


compute_metrics_and_build_table <- function(predicted_classes, actual_labels) {
  # Compute confusion matrix
  cm <- confusionMatrix(data = predicted_classes, reference = actual_labels)
  
  TP <- as.numeric(cm$table[2,2])
  TN <- as.numeric(cm$table[1,1])
  FP <- as.numeric(cm$table[2,1])
  FN <- as.numeric(cm$table[1,2])
  
  TPR = TP / (TP + FN)
  FPR = FP / (FP + TN)
  Precision = TP / (TP + FP)
  Recall = TP / (TP + FN)
  
  MCCoef = ((TP*TN) - (FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  
  Po = (TP + TN) / sum(cm$table)
  Pe1 = ((TP + FN) / sum(cm$table)) * ((TP + FP) / sum(cm$table))
  Pe2 = ((FP + TN) / sum(cm$table)) * ((FN + TN) / sum(cm$table))
  Pe = Pe1 + Pe2
  
  # Extract metrics for each class
  class_metrics <- data.frame(
    Class = rownames(cm$table),
    TP_Rate = round(TPR, digits=3),
    FP_Rate = round(FPR, digits=3),
    Precision = round(Precision, digits=3),
    Recall = round(Recall, digits=3),
    F_Measure = round(2 * Precision * Recall / (Precision + Recall), digits=3),
    MCC = round(MCCoef, digits=3),
    Kappa = round((Po - Pe) / (1 - Pe), digits=3)
  )
  
  # # Compute ROC area for each class
  # ROC_Area <- sapply(predicted_classes, function(class_pred) {
  #   roc_curve <- roc(ifelse(predicted_classes == class_pred, 1, 0), ifelse(actual_labels == class_pred, 1, 0))
  #   auc(roc_curve)
  # })
  
  # # Compute weighted average of each measure
  # weighted_average <- colMeans(class_metrics[, -1])
  # weighted_average["Class"] <- "Weighted Average"
  
  # Bind the class metrics together
  result_table <- rbind(class_metrics) 
  # data.frame(Class = "ROC Area", Value = ROC_Area),
  # weighted_average)
  
  print(cm$table)
  
  # Return the result table
  return(result_table)
}


##############################################
# Split training and testing sets
##############################################
# split dataset
set.seed(31)
split <- initial_split(df, prop = 0.66, strata = Class)
training <- training(split)
test <- testing(split)

balanced.training.df <- bootstrap_balance(training, "Class", 0.5)

##############################################
# Model 1: Naive Bayes -Chuong
##############################################

# Build Naive Bayes model from training set
nb.mdl <-naiveBayes(Class ~ ., data = balanced.training.df)

# Test the model on testing set
nb.mdl.pred <- predict(nb.mdl, newdata = test, type = "class")
nb.mdl.pred.cf <- compute_metrics_and_build_table(nb.mdl.pred, test$Class)
nb.mdl.pred.cf

# Test the model on original testing set
nb.mdl.main.pred <- predict(nb.mdl, newdata = main.test, type = "class")
nb.mdl.main.pred.cf <- compute_metrics_and_build_table(nb.mdl.main.pred, main.test$Class)
nb.mdl.main.pred.cf


##############################################
# Model 2: Logistic Reg -Chuong
##############################################
# disable scientific notation
options(scipen=999)

logitModel <- glm(Class ~ ., data = balanced.training.df, family = "binomial") 

summary(logitModel)

# Predict probability on test set
logitmdl.pred <- predict(logitModel, test, type = "response")
# performance measures on the test dataset
logitmdl.pred <- factor(ifelse(logitmdl.pred >= 0.5, "Y", "N"))

# Test the model on testing set
nb.mdl.pred.cf <- compute_metrics_and_build_table(logitmdl.pred, test$Class)
nb.mdl.pred.cf


# Predict probability on original testing set
logitmdl.main.pred <- predict(logitModel, main.test, type = "response")
# performance measures on the test dataset
logitmdl.main.pred <- factor(ifelse(logitmdl.main.pred >= 0.5, "Y", "N"))
logitmdl.main.pred

# Test the model on testing set
nb.mdl.pred.cf <- compute_metrics_and_build_table(logitmdl.main.pred, 
                                                  main.test$Class)
nb.mdl.pred.cf

##############################################
# Model 3: Decision tree -Chuong
##############################################



##############################################
# Model 4: Random Forest -Brandon
##############################################
# Para Tuning
library(RWeka)
modelLookup("J48")

set.seed(31)
# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)
model1 <- train(Class ~ ., data = train, method = "J48", trControl = train_control)
model1
plot(model1)
test_pred1 <- predict(model1, newdata = test)
confusionMatrix(test_pred1, test$Class, positive = 'Y')
#82.19% accuracy

## use tuneLength
model2 <- train(Class ~ ., data = train, method = "J48", trControl = train_control,
                tuneLength = 4
)
model2
plot(model2)
test_pred2 <- predict(model2, newdata = test)
confusionMatrix(test_pred2, test$Class)
#82.3% accuracy


##############################################
# Model 5: KNN -Brandon
##############################################
# Para Tuning
# use KNN Model
knnModel <- train(Class ~., data = train, method = "knn",
                  trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

test_pred4 <- predict(knnModel, newdata = test)
confusionMatrix(test_pred4, test$Class)
#81.36% accuracy


##############################################
# Model 6: NN -Brandon
##############################################



###############################################################################
# Part III: Model Evaluation & Interpretation
###############################################################################




