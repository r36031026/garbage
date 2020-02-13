library(caret)
library(xgboost)
library(data.table)
library(dplyr)
library(psych)
library(FNN)
train = read.csv('./train.csv')
test = read.csv('./test.csv')
train%>%View()
test%>%View()
### build xgboost as base line ####
set.seed(707)
#### featuring ####
# xgboost of multiple classification need to start from zero 
all = train%>%
  bind_rows(test)%>%
  mutate(Station = as.numeric(as.factor(Station)))%>%
  mutate(County = as.numeric(as.factor(County)))%>%
  mutate(Location = as.numeric(as.factor(Location)))%>%
  rename (shore_line = 海岸段)%>%
  rename(city = 縣市)



# feature engineering 
# back to train and test with all numeric columns
train = all %>% filter(!is.na(LEVEL))
upload = all %>% filter(is.na(LEVEL))
train$LEVEL
knn.reg(train[,c("Seat",'city','Season')], test = upload[,c("Seat",'city','Season')], train[,"LEVEL"], k = 3, algorithm=c("kd_tree"))
knn.reg(train[,c("Seat",'city')], test = upload[,c("Seat",'city')], train[,"LEVEL"], k = 3, algorithm=c("kd_tree"))


#### modeling train--> train/test --> model ####
# Full data set
train_index <- sample(1:nrow(train), nrow(train)*0.75)

#### train and test part 1 ####
# xgboost find best cv 
numberOfClasses <- length(unique(train$LEVEL))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
nround    <- 200 # number of XGBoost rounds
cv.nfold  <- 3

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)
# cv detail
OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = train_label + 1)
# confusion matrix
confusionMatrix(factor(OOF_prediction$max_prob),
                factor(OOF_prediction$label),
                mode = "everything")
OOF_prediction%>%View()
# xgboost model building and testing on the test set 
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)

# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label + 1,
         max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction$max_prob),
                factor(test_prediction$label),
                mode = "everything")
# local leaderboard
temp=cohen.kappa(factor(test_prediction$max_prob)%>%cbind(
  factor(test_prediction$label)))
temp$weighted.kappa
# xgboost importance csv saving 
names <-  colnames(train[,-35])
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
fwrite(importance_matrix,'./importance_xgb_v1.csv')
#### full / upload ####
test = read.csv('./test.csv')
full_model =xgb.train(params = xgb_params,
                      data = full_matrix,
                      nrounds = nround)
upload_pred <- predict(bst_model, newdata = upload_data)
upload_prediction <- matrix(upload_pred, nrow = numberOfClasses,
                            ncol=length(upload_pred)/numberOfClasses) %>%
  t() %>%
  data.frame()%>%
  mutate(LEVEL= max.col(., "last"))%>%
  cbind(test)%>%
  mutate(ID = paste0(Station,'_',Season))%>%
  select(ID,LEVEL)


# ready to upload format 
fwrite(upload_prediction,'./submission_xgboost_v1.csv',row.names = FALSE)
