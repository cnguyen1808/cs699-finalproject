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
library(FSelector)
library(RWeka)


# clean environment
rm(list=ls())

dir <- file.path("C:", "Users", "Admin", "iCloudDrive",
                 "CS699 - Data Mining", "FinalProject", "git")
setwd(dir)

 # setwd("C:\\Users\\brand\\OneDrive\\Documents\\699\\TermProject\\Data")

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

#find highly correlated columns to remove duplicate influence (corr > .7)
corr <- cor(Project_Data_Cleansed2[,-ncol(Project_Data_Cleansed2)])
findCorrelation(corr, cutoff = 0.7, names = TRUE)
CorrColumnRemoved <- findCorrelation(corr, cutoff = 0.7, names = FALSE)

# Apply correlated filtering on train & test data sets
Project_Data_Cleansed2 <- Project_Data_Cleansed2[, -CorrColumnRemoved]
head(Project_Data_Cleansed2)

#Scale the data

scale(Project_Data_Cleansed2[,-ncol(Project_Data_Cleansed2)])
Project_Data_Cleansed2[,-ncol(Project_Data_Cleansed2)] <- scale(Project_Data_Cleansed2[,-ncol(Project_Data_Cleansed2)])

sapply(Project_Data_Cleansed2, class)
head(Project_Data_Cleansed2)


###############################################################################
# Part II: Feature Selection and Imbalanced-data Treatment Methods
###############################################################################

#################################################################
# Split training and testing sets - first splitting
#################################################################
df_ft1 <- Project_Data_Cleansed2

# Split processed data set into training and testing data sets
# This test data set will be used in the very final evaluation step
splitting.df <- df_ft1
splitting.df$Class <- factor(splitting.df$Class, levels = c("Y", "N"))

set.seed(31)
outer.split <- initial_split(splitting.df, prop = 0.8, strata = Class)
main.train.df <- training(outer.split)
main.test.df <- testing(outer.split)

# Project_Data_Cleansed2 <- main.train


###########################
# Feature selection 1 - Boruta
###########################

#Boruta selection function that will remove all rejected functions from the dataset
boruta_selection <- function(df) {
  
  #Use boruta to identify unimportant attributes
  df$Class <- factor(df$Class, 
                      levels = c("Y", "N")) # N -> Y
  borutaResult <- Boruta(Class ~ ., data = df)
  borutaResult$finalDecision
  
  #remove rejected columns
  BorutaRejectedColumns <- names(borutaResult$finalDecision[borutaResult$finalDecision == "Rejected"])
  boruta_df <- df[,!(names(df) %in% BorutaRejectedColumns)]

  return(BorutaRejectedColumns)
}

boruta_selection(main.train.df) # Test of Function

###########################
# Feature selection 2 - Information Gain
###########################
info_gain_selection <- function(df) {
  # information gain
  info.gain <- information.gain(Class ~ ., df)
  info.gain <- cbind(rownames(info.gain), data.frame(info.gain, row.names=NULL))
  names(info.gain) <- c("Attribute", "Info Gain")
  sorted.info.gain <- info.gain[order(-info.gain$`Info Gain`), ]
  sorted.info.gain
  
  #Select columns that correspond to 50% of the total information gain
  # Calculate the total sum of Info Gain
  total_info_gain <- sum(sorted.info.gain$`Info Gain`)
  # Determine 50% of the sum
  target_percent <- 50
  target_sum <- total_info_gain * (target_percent/ 100)
  target_sum
  # Find the number of rows needed to reach the target sum
  selected_rows <- sorted.info.gain %>%
    filter(cumsum(sorted.info.gain$`Info Gain`) <= target_sum)
  selected_rows
  
  # Extract unique attribute names
  selected_attributes <- unique(selected_rows$Attribute)
  selected_attributes
  # Subset your dataframe based on the selected attributes
  info_gain_selection_df <- df[, c(selected_attributes,"Class")]
  
  return(selected_attributes)
  
}

info_gain_selection(main.train.df) #test of information gain selection


###########################
# Feature selection 3
###########################

apply_pca <- function(train_data, test_data, main_train_data, main_test_data) {
  # Apply PCA on the training data set
  pc <- prcomp(train_data[, -ncol(train_data)] %>% 
                 select_if(~ all(is.numeric(.))), 
               center = TRUE, scale = TRUE) # exclude class attribute

  datasets <- list(train_data, test_data, main_train_data, main_test_data)
  names <- c("pca_train_df", "pca_test_df", "pca_main_train_df", "pca_main_test_df")
  pca_dfs <- list()
  
  for (i in seq_along(datasets)) {
    # Map original attributes to new attributes created by PCA
    pca_df <- predict(pc, datasets[[i]])
    pca_df <- data.frame(pca_df, datasets[[i]][, ncol(datasets[[i]])])
    
    # Rename the last column to 'Class'
    colnames(pca_df)[ncol(pca_df)] <- 'Class'
    
    # Select columns to preserve 90% of information
    num_cols_to_select <- as.integer(ncol(pca_df) * 0.9)
    pca_df <- pca_df[, c(1:num_cols_to_select, ncol(pca_df))]
    
    pca_dfs[[i]] <- pca_df
  }
  
  names(pca_dfs) <- names
  
  # Return the transformed training and testing data sets
  return(pca_dfs)
}

# apply_pca(split.train, split.test, main.train.df, main.test.df)


###############################################
# Balance with Oversampling and Undersampling - Chuong
###############################################
###################
# (1) over-sampling 
###################
over_balance <- function(imbalanced.df) {
  
  imbalanced.df <<- imbalanced.df
  
  # Perform over-sampling + under-sampling
  sampled_df <- ROSE::ovun.sample(Class ~ ., 
                                  data = as.data.frame(imbalanced.df),
                                  method = "over", 
                                  p = 0.5,
                                  seed = 31)$data
  
  return(sampled_df)
}


####################
# (2) under-sampling 
####################
under_balance <- function(imbalanced.df) {
  
  imbalanced.df <<- imbalanced.df
  
  # Perform over-sampling + under-sampling
  sampled_df <- ROSE::ovun.sample(Class ~ ., 
                                  data = as.data.frame(imbalanced.df),
                                  method = "under", 
                                  p = 0.5,
                                  seed = 31)$data
  
  return(sampled_df)
}

###########################
# (3) over + under-sampling 
###########################
combine_sampling_balance <- function(imbalanced.df) {
  
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


##################################
# (4) Bootstrap balancing - Chuong
##################################
# Function to perform bootstrap resampling to balance dataset
bootstrap_balance <- function(df) {
  balance_ratio = 1
  
  # Split dataset into majority and minority classes
  majority_class <- df[df[["Class"]] == "N", ]
  minority_class <- df[df[["Class"]] == "Y", ]
  
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



###############################################################################
# Part III: Model Building
###############################################################################

#################################################################
# Split training and testing sets - second splitting
#################################################################
# split data set
set.seed(31)
split <- initial_split(main.train.df, prop = 0.66, strata = Class)
split.train <- training(split)
split.test <- testing(split)



##############################################
# Model 1: Naive Bayes -Chuong
##############################################

# Function to train and test Naive Bayes model
naive.bayes.model <- function(training_data, testing_data, initaldata_test) {
  # Build Naive Bayes model from training set
  nb_model <- naiveBayes(Class ~ ., data = training_data)
  
  # Test the model on provided testing set
  nb_pred <- predict(nb_model, newdata = testing_data, type = "class")
  nb_pred_cf <- compute_metrics_and_build_table(nb_pred, testing_data$Class)
  
  # Test the model on new testing set
  nb_main_pred <- predict(nb_model, newdata = initaldata_test, type = "class")
  nb_main_pred_cf <- compute_metrics_and_build_table(nb_main_pred, initaldata_test$Class)
  
  return(list(inner_split_test = nb_pred_cf, inital_split_test = nb_main_pred_cf))
}


##############################################
# Model 2: Logistic Reg -Chuong
##############################################
# disable scientific notation
options(scipen=999)

# Function to find optimal threshold - based on recall
findOptimalThreshold <- function(predicted_prob, actuals) {
  thresholds <- seq(0, 1, by = 0.01)
  
  recalls <- sapply(thresholds, function(threshold) {
    predicted_classes <- factor(ifelse(predicted_prob >= threshold, "N", "Y"), 
                                levels = c("Y", "N"))
    confusion_matrix <- confusionMatrix(predicted_classes, actuals)$table
    
    TP <- as.numeric(confusion_matrix[1, 1])
    TN <- as.numeric(confusion_matrix[2, 2])
    FP <- as.numeric(confusion_matrix[1, 2])
    FN <- as.numeric(confusion_matrix[2, 1])
    
    TPR <- TP / (TP + FN)
    FPR <- FP / (FP + TN)
    Precision <- TP / (TP + FP)
    Sensitivity <- TPR
    Specificity <- TN / (TN + FP)
    
    # Adjust the weights to prioritize recall and sensitivity differently
    # Here, I'm using equal weights for simplicity
    combined_metric <- 0.2*Sensitivity + 0.75*(Specificity + Sensitivity)
    return(combined_metric)
  })
  optimal_threshold <- thresholds[which.max(recalls)]
  return(optimal_threshold)
}

# Function: logistic regression pipeline 
logistic.regression.model <- function(train, test, 
                                      original_test,
                                      top_vars = 15) {
  # Train first model using the later split train and test data sets
  logitModel1 <- glm(Class ~ ., data = train, family = "binomial")
  summary(logitModel1)
  
  # Predict probability on test set
  logitmdl.pred.prob <- predict(logitModel1, test, type = "response")
  
  # Apply threshold based on recall optimization
  optimal_threshold <- findOptimalThreshold(logitmdl.pred.prob, test$Class)
  
  # performance measures on the test data set
  logitmdl.pred <- factor(ifelse(logitmdl.pred.prob >= optimal_threshold, 
                                 "N", "Y"), 
                          levels = c("Y", "N"))
  
  # Test the model on testing set
  logitmdl.pred.cf <- compute_metrics_and_build_table(logitmdl.pred, test$Class)
  
  ### Extract variable importance
  x <- varImp(logitModel1)
  imp <- data.frame(names = rownames(x), overall = x$Overall)
  
  # Select top N variables from the first model
  if (nrow(imp) < top_vars) {
    top_vars <- nrow(imp)
  }  
  
  top_variables <- imp[order(imp$overall, decreasing = TRUE), ][1:top_vars, 
                                                                "names"]

  new_train_glm <- train[, c("Class", top_variables)]
  new_test_glm <- test[, c("Class", top_variables)]
  
  # Train second Logistic Reg model
  new_logitModel <- glm(Class ~ ., data = new_train_glm, family = "binomial")
  
  # Predict probability on original testing set
  logitmdl_main_pred <- predict(new_logitModel, 
                                original_test[, c("Class", top_variables)], 
                                type = "response")
  
  # performance measures on the original test data set
  logitmdl_main_pred <- factor(ifelse(logitmdl_main_pred >= optimal_threshold, 
                                      "N", "Y"), 
                               levels = c("Y", "N"))
  
  # Test the model on testing set
  logitmdl_main_pred_cf <- compute_metrics_and_build_table(logitmdl_main_pred, 
                                                           original_test$Class)
  
  return(list(
    model1_summary = summary(logitModel1),
    threshold = optimal_threshold,
    model1_performance = logitmdl.pred.cf,
    top_variables = top_variables,
    model2_summary = summary(new_logitModel),
    model2_performance = logitmdl_main_pred_cf
  ))
}

# Usage example:
result <- logistic.regression.model(split.train, split.test, main.test.df)
print("Model 1 Summary:")
print(result$model1_summary)
cat("Optimal Threshold:", result$threshold, "\n")
print("Model 1 Performance:")
print(result$model1_performance)
print("Top Variables Selected:")
print(result$top_variables)
print("Model 2 Summary:")
print(result$model2_summary)
print("Model 2 Performance:")
print(result$model2_performance)



##############################################
# Model 3: Decision tree -Chuong
##############################################

### Function: J48 pipeline
DecisionTree.model <- function(original_train_data, 
                               original_test_data) {

  # pca_train_df <- original_train_data
  # pca_test_df <- original_test_data
  # 
  ### Decision tree model 1: J48 
  # Repeat 10-fold cross-validation 5 times
  j48_train_control <- trainControl(method = "repeatedcv", 
                                    number = 10, repeats = 5,
                                    summaryFunction = defaultSummary)
  
  # Define Grid
  J48Grid <-  expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
  
  # Train J48 model
  j48_model <- train(Class ~ ., 
                     data = original_train_data, 
                     method = "J48", 
                     trControl = j48_train_control,
                     tuneGrid = J48Grid
  )
  
  # Predict on the test set
  j48_test_pred <- predict(j48_model, newdata = original_test_data)
  
  j48_test_pred = factor(j48_test_pred, levels = c("Y", "N"), ordered = TRUE)
  j48_pred_cf <- compute_metrics_and_build_table(j48_test_pred, 
                                                 original_test_data$Class)
  
  
  ### Decision tree model 2: rpart 
  # Train the second model using rpart algorithm
  # Repeat 10-fold cross-validation 5 times
  rpart_train_control <- trainControl(method = "repeatedcv", 
                                      number = 10, repeats = 5, 
                                      summaryFunction = defaultSummary)
  
  
  # Train rpart model
  rpart_model <- train(Class ~ ., 
                       data = original_train_data, 
                       method = "rpart", 
                       trControl = rpart_train_control,
                       tuneLength = 10, 
  )
  
  # Predict on the test set
  rpart_test_pred <- predict(rpart_model, newdata = original_test_data)
  rpart_test_pred = factor(rpart_test_pred, levels = c("Y", "N"), ordered = TRUE)
  rpart_pred_cf <- compute_metrics_and_build_table(rpart_test_pred, 
                                                   original_test_data$Class)
  
  
  return(list(j48_model = j48_model, j48_performance = j48_pred_cf,
              rpart_model = rpart_model, rpart_performance = rpart_pred_cf))
}

# Usage example:
dt_result <- train_j48_with_pca(split.train, split.test, main.train.df, main.test.df)

# Access J48 model and performance
print("J48 Model:")
print(dt_result$j48_model)
print("J48 Performance:")
print(dt_result$j48_performance)

# Access rpart model and performance
print("rpart Model:")
print(dt_result$rpart_model)
print("rpart Performance:")
print(dt_result$rpart_performance)


##############################################
# Model 4: Random Forest -Brandon
##############################################
# Para Tuning
library(RWeka)
# modelLookup("rf")
# test_df <- over_balance(main.train.df)
# test_df <- boruta_selection(main.train.df)
# #split dataset
# set.seed(31)
# split <- initial_split(test_df, prop = 0.66, strata = Class)
# split.train <- training(split)
# split.test <- testing(split)
# 

train_rfModel <- function(train,test,original_test) {
  
  #Establish control parameters
  ctrl <- trainControl(method = "CV",
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       savePredictions = TRUE)
  #number of features considered at each split
  mtryValues <- seq(2, ncol(train)-1, by = 1)
  
  #train random forest model
  rfFit <- caret::train(x = train[, -nrow(train)], 
                        y = train$Class,
                        method = "rf",
                        ntree = 100,
                        tuneGrid = data.frame(mtry = mtryValues),
                        importance = TRUE,
                        metric = "ROC",
                        trControl = ctrl)
  rfFit
  
  ## variable importance
  imp <- varImp(rfFit)
  imp
  
  #Use model to predeict test data
  rf_pred <- predict(rfFit, test)
  rf_performance_pred <- compute_metrics_and_build_table(rf_pred, test$Class)
  #Use model on main test data
  rf_pred <- predict(rfFit, original_test)
  rf_performance <- compute_metrics_and_build_table(rf_pred, original_test$Class)
  
  return(list(rf_model = rfFit, 
              rf_performance_pred = rf_performance_pred,
              rf_performance_main = rf_performance_main ))
}



##############################################
# Model 5: KNN -Brandon
##############################################
# Para Tuning
# use KNN Model
#split dataset
# set.seed(31)
# split <- initial_split(df, prop = 0.66, strata = Class)
# split.train <- training(split)
# split.test <- testing(split)

#Run Feature selected dataset through KNN model
train_and_select_knnModel <- function(train,test,original_test) {
  
  # repeat 10-fold cross-validation 5 times
  knn_train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                                    summaryFunction = defaultSummary)
  
  #Train the initial KNN model on untuned model
  knnModel_untuned <- train(Class ~., data = train, method = "knn",
                            trControl= knn_train_control,
                            preProcess = c("center", "scale"),
                            tuneLength = 200)
  
  #Run Mode on Test dataset
  test_pred_untuned <- predict(knnModel_untuned, newdata = test)
  #Performace statistics of untuned model
  untuned_statistics <- compute_metrics_and_build_table(test_pred_untuned, test$Class)
  
  
  #Training the KNN Model with a Specific Grid of K Values to tune it:
  knnGrid <-  expand.grid(k = seq(1, 100, 2))
  knnModel_tuned <- train(Class ~., data = train, method = "knn",
                          trControl=knn_train_control,
                          preProcess = c("center", "scale"),
                          tuneGrid = knnGrid)
  
  #Run Model on tuned dataset
  test_pred_tuned <- predict(knnModel_tuned, newdata = test)
  #Performace statistics of tuned model
  tuned_statistics_pred <- compute_metrics_and_build_table(test_pred_tuned, test$Class)
  
  #run model on main test data
  test_pred_tuned <- predict(knnModel_tuned, newdata = original_test)
  #Performace statistics of tuned model
  tuned_statistics_main <- compute_metrics_and_build_table(test_pred_tuned, original_test$Class)
  
  return(list(knn_model = knnModel_tuned, 
              knn_performance_pred = tuned_statistics_pred,
              knn_performance_main = tuned_statistics_main))
  
}

#Example

# #Run Pre-Proccessed data through Balancing
# knn_main_train <- over_balance(main.train.df)
# #Run Balanced dataset through Feature Selection
# knn_main_train <- info_gain_selection(knn_main_train)
# #Run KNN model
# train_and_select_knnModel(knn_main_train)

# #balance_functions <- c("over_balance", 
#                        "under_balance", 
#                        "combine_sampling_balance", 
#                        "bootstrap_balance")
# #feature_selection_function <- c("boruta_selection",
#                                 "info_gain_selection",
#                                 "apply_pca")
# 
# '''
# This will run by running our main test Dataset for each balancing function, then
# running each balanced dataset for each type of feature selection:
#    Example: 
#    balance_functions <- c("over_balance", 
#                        "under_balance", 
#                        "combine_sampling_balance", 
#                        "bootstrap_balance")
#    feature_selection_function <- c("boruta_selection",
#                                     "info_gain_selection",
#                                     "apply_pca")
#   for (balance_func in balance_functions){
#     for (feature in feature_selection_function){
#       ### Build model from balanced training set
#       knn.model <- feature(balance_func(main.train.df)))
#       #run model on test dataset
#       
#                               '''


##############################################
# Model 6: NN -Brandon
##############################################
# test_df <- over_balance(main.train.df)
# test_df <- boruta_selection(main.train.df)
# library(rsample)
# set.seed(31)
# split <- initial_split(test_df, prop = 0.66, strata = Class)
# split.train <- training(split)
# split.test <- testing(split)

#nnet Model function
train_nnetModel <- function(train,test,original_test) {
  
  #Establish control parameters
  ctrl <- trainControl(method = "CV", number = 10,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       savePredictions = TRUE)
  ## size: number of units in the hidden layer
  ## decay: parameter for weight decay; also referrred to as L2 regularization
  nnetGrid <- expand.grid(size = 1:5, decay = c(0, .1, 1, 2))
  
  set.seed(31)
  #Run nnet Model
  nnetFit <- train(x = train[, -ncol(train)], 
                   y = train$Class,
                   method = "nnet",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneGrid = nnetGrid,
                   trace = FALSE,
                   maxit = 100,
                   MaxNWts = 1000,
                   trControl = ctrl)
  nnetFit
  #check nnet Config
  nnetFit$bestTune
  
  #run model on test data
  nnet_test_pred <- predict(nnetFit, newdata = test)
  nnet_test_pred
  #check test data performance
  nnet_performance_pred <-compute_metrics_and_build_table(nnet_test_pred, test$Class)
  
  #run on main test dataset
  nnet_test_pred <- predict(nnetFit, newdata = original_test)
  nnet_performance_main <-compute_metrics_and_build_table(nnet_test_pred, original_test$Class)
  
  #return model and model performance
  return(list(nnet_model = nnetFit, 
              nnet_performance_pred = nnet_performance_pred,
              nnet_performance_main = nnet_performance_main))
}

#Example
# train_nnetModel(test_df)



###############################################################################
# Part III: Model Evaluation & Interpretation
###############################################################################

##############################################
# Computing Evaluation Metrics
##############################################

# Define a function to compute metrics and build the result table
compute_metrics_and_build_table <- function(predicted_classes, actual_labels) {
  # Compute confusion matrix
  cm <- confusionMatrix(data = predicted_classes, reference = actual_labels)
  cm_inverted <- confusionMatrix(data = actual_labels, reference = predicted_classes)  # Inverted confusion matrix
  
  class_metrics <- data.frame(
    Class = c("Class 1 (Y = Positive)", "Class 2 (N = Positive)"),
    Acc_Rate = numeric(2),
    TP_Rate = numeric(2),
    FP_Rate = numeric(2),
    Precision = numeric(2),
    Recall = numeric(2),
    F_Measure = numeric(2),
    MCC = numeric(2),
    Kappa = numeric(2),
    ROC_Area = numeric(2)  # Adding ROC area column
  )
  
  for (i in 1:2) {
    if (i == 1) {
      cm_iter <- cm
    } else {
      cm_iter <- cm_inverted
    }
    
    
    TP <- as.numeric(cm_iter$table[1, 1])
    TN <- as.numeric(cm_iter$table[2, 2])
    FP <- as.numeric(cm_iter$table[1, 2])
    FN <- as.numeric(cm_iter$table[2, 1])
    
    Accuracy <- (TP + TN)/sum(as.numeric(cm_iter$table))
    TPR <- TP / (TP + FN)
    FPR <- FP / (FP + TN)
    Precision <- TP / (TP + FP)
    Recall <- TPR
    F_Measure <- 2 * Precision * Recall / (Precision + Recall)
    MCCoef <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # Calculate Kappa inside the loop
    Po <- (TP + TN) / sum(cm_iter$table)
    Pe1 <- ((TP + FN) / sum(cm_iter$table)) * ((TP + FP) / sum(cm_iter$table))
    Pe2 <- ((FP + TN) / sum(cm_iter$table)) * ((FN + TN) / sum(cm_iter$table))
    Pe <- Pe1 + Pe2
    Kappa <- (Po - Pe) / (1 - Pe)
    
    # Calculate ROC area
    # cat("\n", "predicted_classes: ", predicted_classes, "\n")
    # cat("\n", "actual_labels: ", levels(predicted_classes), "\n")
    
    # roc_curve <- roc(ifelse(predicted_classes == 
    #                           levels(predicted_classes)[i], 1, 0),
    #                  ifelse(actual_labels == levels(actual_labels)[i], 1, 0))
    roc_curve <- roc(factor(actual_labels, ordered = TRUE), 
                     factor(predicted_classes, ordered = TRUE))
    roc_area <- auc(roc_curve)
    
    class_metrics[i, "Acc_Rate"] <- round(Accuracy, digits = 3)
    class_metrics[i, "TP_Rate"] <- round(TPR, digits = 3)
    class_metrics[i, "FP_Rate"] <- round(FPR, digits = 3)
    class_metrics[i, "Precision"] <- round(Precision, digits = 3)
    class_metrics[i, "Recall"] <- round(Recall, digits = 3)
    class_metrics[i, "F_Measure"] <- round(F_Measure, digits = 3)
    class_metrics[i, "MCC"] <- round(MCCoef, digits = 3)
    class_metrics[i, "Kappa"] <- round(Kappa, digits = 3)
    class_metrics[i, "ROC_Area"] <- round(roc_area, digits = 3)  # Add ROC area to class_metrics
  }
  
  # Calculate the weighted average of each measure
  weighted_average <- colMeans(class_metrics[, -1])
  weighted_average["Class"] <- "Weighted Average"
  weighted_average <- c("Weighted Average", 
                        weighted_average[-length(weighted_average)])
  
  # Combine class metrics and weighted average
  result_table <- rbind(class_metrics, weighted_average)
  
  return(result_table)
}


##############################################
# Execute all Classification Pipelines
##############################################

### Define the list of all dataset balancing methods
balance_functions <- c("over_balance", 
                       "under_balance", 
                       "combine_sampling_balance", 
                       "bootstrap_balance")


### Define lists to store results of each Classification
NB.results <- list()
LOG.results <- list()
DT.results <- list()
RF.results <- list()
KNN.results <- list()
NN.results <- list()

### Apply each balanced data set to all Classifications
for (balance_func in balance_functions) {
  balanced_training <- match.fun(balance_func)(split.train)
  
  # Apply Boruta on training & testing data sets
  boruta.rejected.cols <- boruta_selection(balanced_training)
  boruta.training <- balanced_training[,!(names(balanced_training) %in% boruta.rejected.cols)]
  boruta.testing <- split.test[,!(names(split.test) %in% boruta.rejected.cols)]
  boruta.main.training <- main.train.df[,!(names(main.train.df) %in% boruta.rejected.cols)]
  boruta.main.testing <- main.test.df[,!(names(main.test.df) %in% boruta.rejected.cols)]
    
  # Apply information gain on training & testing data sets
  infogain.selected.cols <- info_gain_selection(balanced_training)
  infogain.training <- balanced_training[, c(infogain.selected.cols,"Class")]
  infogain.testing <- split.test[, c(infogain.selected.cols,"Class")]
  infogain.main.training <- main.train.df[, c(infogain.selected.cols,"Class")]
  infogain.main.testing <- main.test.df[, c(infogain.selected.cols,"Class")]
    
  # Apply PCA on training & testing data sets
  pca.dfs <- apply_pca(balanced_training, split.test, main.train.df, main.test.df)
  pca.training <- pca.dfs$pca_train_df
  pca.testing <- pca.dfs$pca_train_df
  pca.main.training <- pca.dfs$pca_main_train_df
  pca.main.testing <- pca.dfs$pca_main_test_df
  
  
  # Factor "Class" variable in all datasets
  factor_class <- function(dataset) {
    dataset$Class <- factor(dataset$Class, levels = c("Y", "N"))
    return(dataset)
  }  
  
  balanced_training <- factor_class(balanced_training)
  split.test <- factor_class(split.test)
  main.train.df <- factor_class(main.train.df)
  main.test.df <- factor_class(main.test.df)
  boruta.training <- factor_class(boruta.training)
  boruta.testing <- factor_class(boruta.testing)
  boruta.main.training <- factor_class(boruta.main.training)
  boruta.main.testing <- factor_class(boruta.main.testing)
  infogain.training <- factor_class(infogain.training)
  infogain.testing <- factor_class(infogain.testing)
  infogain.main.training <- factor_class(infogain.main.training)
  infogain.main.testing <- factor_class(infogain.main.testing)
  pca.training <- factor_class(pca.training)
  pca.testing <- factor_class(pca.testing)
  pca.main.training <- factor_class(pca.main.training)
  pca.main.testing <- factor_class(pca.main.testing)
  
  ### Naive Bayes
  # plain
  NB.model <- naive.bayes.model(training_data = balanced_training,
                                     testing_data = split.test,
                                     initaldata_test = main.test.df)
  # boruta
  NB.boruta.model <- naive.bayes.model(training_data = boruta.training,
                                     testing_data = boruta.testing,
                                     initaldata_test = boruta.main.testing)
  # information gain
  NB.infogain.model <- naive.bayes.model(training_data = infogain.training,
                                     testing_data = infogain.testing,
                                     initaldata_test = infogain.main.testing)
  # pca
  NB.pca.model <- naive.bayes.model(training_data = pca.training,
                                     testing_data = pca.testing,
                                     initaldata_test = pca.main.testing)

  NB.results[[balance_func]]$plain_test <- NB.model
  NB.results[[balance_func]]$boruta_test <- NB.boruta.model
  NB.results[[balance_func]]$infogain_test <- NB.infogain.model
  NB.results[[balance_func]]$pca_test <- NB.pca.model


  ### LOGIT REG

  # plain
  LOG.model <- logistic.regression.model(balanced_training,
                                         split.test,
                                         main.test.df)
  # boruta
  LOG.boruta.model <- logistic.regression.model(boruta.training,
                                                boruta.testing,
                                                boruta.main.testing)
  # information gain
  LOG.infogain.model <- logistic.regression.model(infogain.training,
                                                  infogain.testing,
                                                  infogain.main.testing)
  # pca
  LOG.pca.model <- logistic.regression.model(pca.training,
                                             pca.testing,
                                             pca.main.testing)

  LOG.results[[balance_func]]$plain_test <- LOG.model[length(LOG.model)]
  LOG.results[[balance_func]]$boruta_test <- LOG.boruta.model[length(LOG.model)]
  LOG.results[[balance_func]]$infogain_test <- LOG.infogain.model[length(LOG.model)]
  LOG.results[[balance_func]]$pca_test <- LOG.pca.model[length(LOG.model)]

  ### DECISION TREE
  # Plain
  DT.models <- DecisionTree.model(main.train.df, main.test.df)
  # boruta
  DT.boruta.models <- DecisionTree.model(boruta.main.training, boruta.main.testing)
  # information gain
  DT.infogain.models <- DecisionTree.model(infogain.main.training, infogain.main.testing)
  # PCA
  DT.pca.models <- DecisionTree.model(pca.main.training, pca.main.testing)

  DT.results[[balance_func]]$plain_test <- DT.models[c(2,4)]
  DT.results[[balance_func]]$boruta_test <- DT.boruta.models[c(2,4)]
  DT.results[[balance_func]]$infogain_test <- DT.infogain.models[c(2,4)]
  DT.results[[balance_func]]$pca_test <- DT.pca.models[c(2,4)]
  # 
  ### RANDOM FOREST
  # Plain
  RF.models <- train_rfModel(balanced_training, split.test, main.test.df)
  # boruta
  RF.boruta.models <- train_rfModel(balanced_training, split.test, main.test.df)
  # information gain
  RF.infogain.models <- train_rfModel(balanced_training, split.test, main.test.df)
  # PCA
  RF.pca.models <- train_rfModel(balanced_training, split.test, main.test.df)
  
  RF.results[[balance_func]]$plain_test <- RF.models[length(RF.models)]
  RF.results[[balance_func]]$boruta_test <- RF.boruta.models[length(RF.models)]
  RF.results[[balance_func]]$infogain_test <- RF.infogain.models[length(RF.models)]
  RF.results[[balance_func]]$pca_test <- RF.pca.models[length(RF.models)]
  
  
  ### KNN
  # Plain
  KNN.models <- train_and_select_knnModel(balanced_training, split.test, main.test.df)
  # boruta
  KNN.boruta.models <- train_and_select_knnModel(balanced_training, split.test, main.test.df)
  # information gain
  KNN.infogain.models <- train_and_select_knnModel(balanced_training, split.test, main.test.df)
  # PCA
  KNN.pca.models <- train_and_select_knnModel(balanced_training, split.test, main.test.df)
  
  KNN.results[[balance_func]]$plain_test <- KNN.models[length(KNN.models)]
  KNN.results[[balance_func]]$boruta_test <- KNN.boruta.models[length(KNN.models)]
  KNN.results[[balance_func]]$infogain_test <- KNN.infogain.models[length(KNN.models)]
  KNN.results[[balance_func]]$pca_test <- KNN.pca.models[length(KNN.models)]
  
  
  ### NNet
  # Plain
  NN.models <- train_and_select_knnModel(balanced_training, split.test, main.test.df)
  # boruta
  NN.boruta.models <- train_and_select_knnModel(balanced_training, split.test, main.test.df)
  # information gain
  NN.infogain.models <- train_and_select_knnModel(balanced_training, split.test, main.test.df)
  # PCA
  NN.pca.models <- train_and_select_knnModel(balanced_training, split.test, main.test.df)
  
  NN.results[[balance_func]]$plain_test <- NN.models[length(NN.models)]
  NN.results[[balance_func]]$boruta_test <- NN.boruta.models[length(NN.models)]
  NN.results[[balance_func]]$infogain_test <- NN.infogain.models[length(NN.models)]
  NN.results[[balance_func]]$pca_test <- NN.pca.models[length(NN.models)]
  
}


# Access results for each balancing function
for (balance_func in balance_functions) {
  cat("\n", "Balancing method: ", balance_func, "\n")
  
  cat("Naive Bayes with ", balance_func, ":\n")
  print(NB.results[[balance_func]])
  
  cat("Logistic Regression with", balance_func, ":\n")
  print(LOG.results[[balance_func]])
  
  cat("Decision Tree with", balance_func, ":\n")
  print(DT.results[[balance_func]])
}


