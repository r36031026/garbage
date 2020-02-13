library(caret)
library(xgboost)
library(data.table)
library(dplyr)
library(psych)

train = read.csv('./train revise.csv')
test = read.csv('./test revise.csv')
### build xgboost as base line ####
set.seed(707)
#### featuring ####
# xgboost of multiple classification need to start from zero 
all = train%>%
  bind_rows(test)%>%
  # separate the station part for EN and NUMERIC part for KNN 
  mutate(station_num = gsub("[^0-9.-]","",Station),
         station_num = as.numeric(station_num))%>%
  mutate(station_EN = gsub("[^A-Z]","",Station),
         station_EN = as.numeric(station_EN))%>%
  mutate(Station = as.numeric(as.factor(Station)))%>%
  mutate(County = as.numeric(as.factor(County)))%>%
  mutate(Location = as.numeric(as.factor(Location)))%>%
  # make the Seat simple version 
  mutate(Seat_simple = case_when(Seat%in%c(1,2,8)~ 'north',
                                 Seat%in%c(2,3,4)~ 'west',
                                 Seat%in%c(4,5,6)~ 'south',
                                 Seat%in%c(6,7,8)~ 'east'))%>%
  mutate(Seat_simple = as.numeric(Seat_simple))

# some mapping table of mean and sd with level info 
City_season = train%>%
  group_by(縣市,Season)%>%
  summarise(mean_city_season = mean(LEVEL,na.rm= T),
         sd_city_season = sd(LEVEL,na.rm =T))%>%
  ungroup()

City= train%>%  
  group_by(縣市)%>%
  summarise(mean_county = mean(LEVEL,na.rm= T),
         sd_city = sd(LEVEL,na.rm =T),
         max_city = max(LEVEL),
         min_city = min(LEVEL),
         median_city = median(LEVEL)
  )%>%
  ungroup()

Season_seat = train%>%  group_by(Season,Seat)%>%
  summarise(mean_season_seat = mean(LEVEL,na.rm = T),
         sd_season_seat = sd(LEVEL,na.rm =T),
         max_season_seat = max(LEVEL),
         min_season_seat = min(LEVEL),
         median_season_seat = median(LEVEL)
  )%>%
  ungroup()
  
Seat = train%>%
  group_by(Seat)%>%
  summarise(mean_seat = mean(LEVEL,na.rm = T),
         sd_seat = sd(LEVEL,na.rm = T),
         max_seat = max(LEVEL),
         min_seat = min(LEVEL),
         median_seat = median(LEVEL))%>%
  ungroup()

CSS = train%>%
  group_by(縣市,Season,Seat)%>%
  summarise(mean_CSS = mean(LEVEL,na.rm = T),
         sd_CSS = sd(LEVEL,na.rm = T),
         max_CSS = max(LEVEL),
         min_CSS = min(LEVEL),
         median_CSS = median(LEVEL)
  )

all = all%>%
  left_join(City)%>%
  left_join(City_season)%>%
  left_join(CSS)%>%
  left_join(Seat)

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
param <- list("objective" = "reg:linear", 
              "eval_metric" = "mae",
              'colsample_bytree' = 0.7,
              'subsample' = 0.7,
              'max_depth' = 6,
              'eta' = 0.1,
              'seed' = 12345)
nround    <- 200 # number of XGBoost rounds
xgb_model = xgboost(param = param,  data = train_matrix, nrounds = 200, print_every_n = 10)
train_pred = predict(xgb_model,test_matrix)

# confusion matrix
confusionMatrix(factor(round(train_pred)),
                factor(round(test_label)),
                mode = "everything")

# local leaderboard
temp=cohen.kappa(
  factor(round(train_pred))%>%
           cbind(
  factor(round(test_label))))
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names(train_data), model = xgb_model)
fwrite(importance_matrix,'./importance_xgb_regv1.csv')
#### full / upload ####
test = read.csv('./test.csv')
nround    <- 200 # number of XGBoost rounds
xgb_model = xgboost(param = param,  data = full_matrix, nrounds = 200, print_every_n = 10)
test$LEVEL= round(predict(xgb_model,upload_data))
# write
test%>% 
  mutate(ID = paste0(Station,'_',Season))%>%
  select(ID,LEVEL)%>%
  fwrite('./submission_xgboost_reg_v1.csv',row.names = FALSE)
