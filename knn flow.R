library(caret)
library(xgboost)
library(data.table)
library(dplyr)
library(psych)
train = read.csv('./train.csv')
test = read.csv('./test.csv')
### build xgboost as base line ####
set.seed(707)
#### featuring ####
# xgboost of multiple classification need to start from zero 
all = train%>%
  bind_rows(test)%>%
  mutate(Station = as.numeric(as.factor(Station)))%>%
  mutate(County = as.numeric(as.factor(County)))%>%
  mutate(Location = as.numeric(as.factor(Location)))%>%
  mutate(LEVEL = LEVEL-1)%>%
  rename (shore_line = 海岸段)%>%
  rename(city = 縣市)

all%>%
  # separate the station part for EN and NUMERIC part for KNN 
  mutate(station_num = gsub("[^0-9.-]","",Station),
         station_num = as.numeric(station_num))%>%
  mutate(station_EN = gsub("[^A-Z]","",Station))%>%
  # make the Seat simple version 
  mutate(Seat_simple = case_when(Seat%in%c(1,2,8)~ 'north',
                                 Seat%in%c(2,3,4)~ 'west',
                                 Seat%in%c(4,5,6)~ 'south',
                                 Seat%in%c(6,7,8)~ 'east'))
# some mapping table of mean and sd with level info 
a = train%>%
  group_by(County,Season)%>%
  mutate(mean_county_season = mean(LEVEL,na.rm= T),
         sd_county_season = sd(LEVEL,na.rm =T))%>%
  ungroup()%>%
  group_by(County)%>%
  mutate(mean_county = mean(LEVEL,na.rm= T),
         sd_county = sd(LEVEL,na.rm =T),
         max_county = max(LEVEL),
         min_county = min(LEVEL),
         median_county = median(LEVEL)
         )%>%
  ungroup()%>%
  group_by(Season,Seat)%>%
  mutate(mean_season_seat = mean(LEVEL,na.rm = T),
         sd_season_seat = sd(LEVEL,na.rm =T),
         max_season_seat = max(LEVEL),
         min_season_seat = min(LEVEL),
         median_season_seat = median(LEVEL)
  )%>%
  ungroup()%>%
  group_by(Seat)%>%
  mutate(mean_seat = mean(LEVEL,na.rm = T),
         sd_seat = sd(LEVEL,na.rm = T),
         max_seat = max(LEVEL),
         min_seat = min(LEVEL),
         median_seat = median(LEVEL)
  )%>%
  ungroup()%>%
  group_by(County,Season,Seat)%>%
  mutate(mean_CSS = mean(LEVEL,na.rm = T),
         sd_CSS = sd(LEVEL,na.rm = T),
         max_CSS = max(LEVEL),
         min_CSS = min(LEVEL),
         median_CSS = median(LEVEL)
  )
  a%>%View()
train%>%names()
  
# feature engineering 
# back to train and test with all numeric columns
train = all %>% filter(!is.na(LEVEL))
upload = all %>% filter(is.na(LEVEL))


#### modeling train--> train/test --> model ####
# Full data set
train_index <- sample(1:nrow(train), nrow(train)*0.75)
data_variables <- as.matrix(train[,-35])
data_label <- train[,"LEVEL"]
data_matrix <- xgb.DMatrix(data = as.matrix(train), label = data_label)
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)
# full data set 
full_matrix = xgb.DMatrix(data = data_variables,label = data_label)
# upload date 
upload_data <- xgb.DMatrix(data =as.matrix(upload[-35]))

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
