library(caret)
library(xgboost)
library(data.table)
library(dplyr)
library(psych)

train = read.csv('./train revise.csv')
train%>%nrow()
test = read.csv('./test revise.csv')
test%>%nrow()
### build xgboost as base line ####
set.seed(707)
#### featuring ####
# xgboost of multiple classification need to start from zero 


all = train%>%
  bind_rows(test)%>%
  # separate the station part for EN and NUMERIC part for KNN 
  mutate(station_EN = gsub("[^A-Z]","",Station),
         station_EN = as.factor(station_EN))%>%
  mutate(seat_season_combo = as.factor(paste0(Season,Seat)))%>%
  mutate(seat_shore_combo = as.factor(paste0(海岸段,Seat)))%>%
  mutate(seat_Region_combo = as.factor(paste0(Region,Seat)))%>%
  # make the Seat simple version 
  # mutate(Seat_simple = case_when(Seat%in%c(1,2,8)~ 'north',
  #                                Seat%in%c(2,3,4)~ 'west',
  #                                Seat%in%c(4,5,6)~ 'south',
  #                                Seat%in%c(6,7,8)~ 'east'))%>%
  # mutate(Seat_and_season  = case_when(Seat_simple == 'north'& Season %in%c(4)~ 2,
  #                                     Seat_simple == 'north' & Season %in% c(3)~1,
  #                                     Seat_simple == 'south' & Season %in% c(1)~1,
  #                                     Seat_simple == 'south' & Season %in% c(2)~2,
  #                                     Seat_simple == 'west' & Season %in% c(1)~1,
  #                                     Seat_simple == 'west' & Season %in% c(2)~2,
  #                                     TRUE~as.numeric(0)))%>%
  # mutate(Seat_simple = as.factor(Seat_simple))%>%
  # top 3 appear and top 3 biggest  
 # mutate(top_3_appear = Foam.material+Fishing.nets.and.ropes+Plastic.bottle.container)%>%
#  mutate(top_3_biggest = Fishing.nets.and.ropes+ Float +Others ) %>%
  # group_by(Season,Seat)%>%
  # mutate(ss_Foam_mean = mean(Foam.material,na.rm=T),
  #        ss_Fishingnr_mean = mean(Fishing.nets.and.ropes),
  #        ss_Plasticbc_mean = mean(Plastic.bottle.container))%>%
  ungroup()%>%
#  arrange(Location,Season)%>%
#  fill(everything())%>%fill(everything(),.direction = 'down')%>%
  mutate(type_of_shore_by_sea_strongess = X1暴露岩岸*10 + X2暴露人造結構物 *9 + X3暴露岩盤*8+
           X4沙灘*7+X5砂礫混合灘*6+X6礫石灘*5+X7開闊潮間帶*4+X8遮蔽岩岸*3+X9遮蔽潮間帶*2+X10遮蔽濕地*1)%>%
  mutate(number_of_shore_type = X1暴露岩岸 + X2暴露人造結構物  + X3暴露岩盤+
           X4沙灘+X5砂礫混合灘+X6礫石灘+X7開闊潮間帶+X8遮蔽岩岸+X9遮蔽潮間帶+X10遮蔽濕地)%>%
  mutate(final_type_shore_by_sea_strongness = type_of_shore_by_sea_strongess/number_of_shore_type)%>%
  select(-c(type_of_shore_by_sea_strongess))%>%
 mutate(total_kind_garbage = Fishing.equipment+Fishing.nets.and.ropes+Metal+Foam.material+Paper+Plastic.bottle.container+Plastic.bag+Glass.jar+Others+Cigarette.and.lighter+Float)%>%
 mutate(east = ifelse(Region%in%c(2,5),1,0))%>%
  # most polluted sea shore in Taiwan take up 50% of trash 
  left_join(
    train%>%
      bind_rows(test)%>%
      filter_all(any_vars(grepl(pattern="(瑞芳|大城|青草|分洪道|六輕|白玉|白水|濆水|慶安北路|通宵|竹圍|國聖|布袋)", .)))%>%
      mutate(most_polluted_13 = 1)%>%
      select(County,Location,most_polluted_13)
  )%>%
  replace_na(list(most_polluted_13=0))%>%
  mutate(Station = as.factor(Station))%>%
  mutate(County = as.factor(County))%>%
  mutate(Location = as.factor(Location))

# 海岸段 season seat combination level
ssc_rank = all%>%
  filter(!is.na(LEVEL))%>%
  group_by(Location)%>%
  arrange(LEVEL)%>%
  mutate(rank_sl = row_number())%>%
  ungroup()%>%
  select(Location,Season,rank_sl)

# some mapping table of mean and sd with level info 
Location_LEVEL = all%>%
  filter(!is.na(LEVEL))%>%
  group_by(Location)%>%
  summarise(location_ml = mean(LEVEL))

all%>%nrow()
all = all%>%
  left_join(Location_LEVEL)%>%
  left_join(ssc_rank)



correlationMatrix <- cor(all%>%)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)


# feature engineering 
# back to train and test with all numeric columns
train = all %>% filter(LEVEL>=0)
upload = all %>% filter(is.na(LEVEL))
train%>%View()
upload%>%View()
all%>%nrow()
train%>%nrow()
upload%>%nrow()
# caret training version
xgb_grid_1 = expand.grid(
  nrounds = c(300,500),
  eta = c(0.1,0.3 ),
  max_depth = c(4, 6),
  gamma = 0.1,
  colsample_bytree =c(0.7),
  min_child_weight =c(0.3,0.1),
  subsample=c(0.7,0.5)
  
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 3,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  allowParallel = TRUE
)

# train the model for each parameter combination in the grid,
#   using CV to evaluate
xgb_train_1 = train(
  x = data.matrix(train %>%
                  select(-LEVEL)),
  y = train$LEVEL,
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree"
)
plot(xgb_train_1)

upload$LEVEL= predict(xgb_train_1,newdata = data.matrix(upload)  )
 upload%>%
   mutate(LEVEL=round(LEVEL))%>%
   mutate(ID = paste0(Station,'_',Season))%>%
  select(ID,LEVEL)%>%
  arrange(ID)%>%
  fwrite('./xgboost_reg_v5_tuning_grid.csv',row.names = FALSE)

upload%>%View()









#### modeling train--> train/test --> model ####
# Full data set
train_index <- sample(1:nrow(train), nrow(train)*0.75)
data_variables <- data.matrix(train[,-35])
data_label <- data.matrix(train[,"LEVEL"])
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index,]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)
# full data set 
full_matrix = xgb.DMatrix(data = data_variables,label = data_label)
# upload date 
upload_data <- xgb.DMatrix(data =data.matrix(upload[-35]))

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
temp
#### full / upload ####
test = read.csv('./test.csv')
nround    <- 300 # number of XGBoost rounds
xgb_model = xgboost(param = param,  data = full_matrix, nrounds = 200, print_every_n = 10)
test$LEVEL= round(predict(xgb_model,upload_data))
