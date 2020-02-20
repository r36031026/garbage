library(caret)
library(xgboost)
library(data.table)
library(dplyr)
library(psych)
library(tidyr)

train = read.csv('../train-revise.csv')
test = read.csv('../test-revise.csv')
#### most polluted 13 ####
most_polluted13 = train%>%
  bind_rows(test)%>%
  filter_all(any_vars(grepl(pattern="(瑞芳|大城|青草|分洪道|六輕|白玉|白水|濆水|慶安北路|通宵|竹圍|國聖|布袋)", .)))%>%
  mutate(most_polluted_13 = 1)%>%
  select(County,Location,Season,most_polluted_13)


#### fill the NA of place ####
fill_place = train%>%
  bind_rows(test)%>%
  select(1:22)%>%
  arrange(Station,Season)%>%
  fill(everything())%>%fill(everything(),.direction = 'down')%>%
  fill(everything())%>%fill(everything(),.direction = 'up')

fill_place
### build xgboost as base line ####
set.seed(707)
#### featuring ####
# xgboost of multiple classification need to start from zero 
all = train%>%
  bind_rows(test)%>%
  select(-c(13:22))%>%
  left_join(fill_place)%>%
  left_join(most_polluted13)%>%
  replace_na(list(most_polluted_13=0))%>%
  # separate the station part for EN and NUMERIC part for KNN 
  # mutate(station_num = gsub("[^0-9.-]","",Station),
  #         station_num = as.numeric(station_num))%>%
  mutate(station_EN = gsub("[^A-Z]","",Station),
         station_EN = as.factor(station_EN))%>%
  # make the Seat simple version 
  mutate(top_13_sea_shore_direction  = ifelse(海岸段%in%c(1,3),1,0))%>%
  rename(sea_shore = 海岸段)%>%
  mutate(type_of_shore_by_sea_strongess = X1暴露岩岸*10 + X2暴露人造結構物 *9 + X3暴露岩盤*8+
           X4沙灘*7+X5砂礫混合灘*6+X6礫石灘*5+X7開闊潮間帶*4+X8遮蔽岩岸*3+X9遮蔽潮間帶*2+X10遮蔽濕地*1)%>%
  mutate(number_of_shore_type = X1暴露岩岸 + X2暴露人造結構物  + X3暴露岩盤+
           X4沙灘+X5砂礫混合灘+X6礫石灘+X7開闊潮間帶+X8遮蔽岩岸+X9遮蔽潮間帶+X10遮蔽濕地)%>%
  mutate(final_type_shore_by_sea_strongness = type_of_shore_by_sea_strongess/number_of_shore_type)%>%
  mutate(final_type_shore_by_sea_strongness = ifelse(is.na(final_type_shore_by_sea_strongness),0,final_type_shore_by_sea_strongness))%>%
  select(-c(type_of_shore_by_sea_strongess))%>%
  mutate(top_3_place = ifelse(X1暴露岩岸+X2暴露人造結構物+X4沙灘+X5砂礫混合灘+X6礫石灘>=2,1,0))%>%
  mutate(place_order_one_column = as.factor(case_when(X1暴露岩岸==1~'X1',
                                                      X2暴露人造結構物==1~'X2',
                                                      X3暴露岩盤==1~'X3',
                                                      X4沙灘==1~'X4',
                                                      X5砂礫混合灘==1~'X5',
                                                      X6礫石灘==1~'X6',
                                                      X7開闊潮間帶==1~'X7',
                                                      X8遮蔽岩岸==1~'X8',
                                                      X9遮蔽潮間帶==1~'X9',
                                                      X10遮蔽濕地==1~'X10',
                                                      TRUE~'X0'
  )))%>%
  mutate(stationEN_place = as.factor(paste0(station_EN,place_order_one_column)))%>%
  # make it factor for xgboost 
  mutate(Station = as.factor(Station))%>%
  mutate(County = as.factor(County))%>%
  mutate(Location = as.factor(Location))

# remove the bad features

sapply(all, function(x) sum(is.na(x)))
all%>%glimpse()


# the train and upload split back
train = all %>% filter(!is.na(LEVEL))
upload = all %>% filter(is.na(LEVEL))
train_label = train$LEVEL
upload_label = upload$LEVEL

train = train%>%select(-LEVEL)
upload = upload%>%select(-LEVEL)

#### modeling train--> train/test --> model ####
# Full data set
train_index <- sample(1:nrow(train), nrow(train)*0.75)

# split train data and make xgb.DMatrix


trainMatrix<-sparse.model.matrix(~.-1,data=train[train_index,])
model = xgboost(trainMatrix,label = train_label[train_index],nrounds=300)

# split test data and make xgb.DMatrix
testMatrix<-sparse.model.matrix(~.-1,data=train[-train_index,])
testMatrix
# full data set 
fullMatrix = sparse.model.matrix(~.-1,data=train)
fullMatrix
# upload date 
uploadMatrix = sparse.model.matrix(~.-1,data=upload)
uploadMatrix
#### train and test part 1 ####
# xgboost find best cv 
param <- list("objective" = "reg:linear", 
              "eval_metric" = "mae",
              'colsample_bytree' = 0.7,
              'subsample' = 0.5,
              'max_depth' = 6,
              'eta' = 0.1,
              'seed' = 12345,
              'min_child_wright'=0.3
)
nround    <- 300 # number of XGBoost rounds

test = predict(model,newdata = testMatrix)
test
# # confusion matrix
# confusionMatrix(factor(round(test)),
#                 factor(round(train_label[-train_index])),
#                 mode = "everything")

# local leaderboard
temp=cohen.kappa(
  factor(round(test))%>%
    cbind(
      factor(round(train_label[-train_index]))))
temp$weighted.kappa

#### full / upload ####
test = read.csv('./test.csv')
nround    <- 300 # number of XGBoost rounds
xgb_model = xgboost(data = fullMatrix, label = train_label,print_every_n = 10,nrounds = 300)
xgb.importance(feature_names = names(uploadMatrix), model = xgb_model, trees = NULL,
               data = NULL, label = NULL, target = NULL)

test$LEVEL= round(predict(xgb_model,uploadMatrix))
library(Matrix)
iris%>%glimpse
trainMatrix<-sparse.model.matrix(Species~.,data=iris%>%mutate(Species2=Species))
xgboost(trainMatrix,label = iris$Species,nrounds=100)
test%>%
  mutate(LEVEL=round(LEVEL))%>%
  mutate(ID = paste0(Station,'_',Season))%>%
  select(ID,LEVEL)%>%
  write.csv('./0220_xgboost.csv',row.names = F)
