library(caret)
library(xgboost)
library(data.table)
library(dplyr)
library(psych)
library(gstat)
library(sp)

train = read_csv('./train revise.csv')
test = read_csv('./test revise.csv')

train%>%View()
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
         station_EN = as.numeric(as.factor(station_EN)))%>%
  mutate(Station = as.numeric(as.factor(Station)))%>%
  mutate(County = as.numeric(as.factor(County)))%>%
  mutate(Location = as.numeric(as.factor(Location)))%>%
  # make the Seat simple version 
  mutate(Seat_simple = case_when(Seat%in%c(1,2,8)~ 'north',
                                 Seat%in%c(2,3,4)~ 'west',
                                 Seat%in%c(4,5,6)~ 'south',
                                 Seat%in%c(6,7,8)~ 'east'))%>%
  mutate(Seat_simple = as.numeric(as.factor(Seat_simple)))
# season 1~4
ven = function(i){
  s1 = all%>%
  filter(Season == i) %>%
  select(Lat,Lon,LEVEL)%>%
  filter(!is.na(LEVEL))

total_grid = all%>%
  filter(Season == i)%>%
#  filter(is.na(LEVEL))%>%
  select(Lat,Lon)

coordinates(s1) = ~Lon+Lat
coordinates(total_grid) = ~Lon+Lat
#plot(s1)
#plot(total_grid)
#bbox(s1)
#bbox(total_grid)

lzn.vgm <- variogram(LEVEL~1, s1) # calculates sample variogram values 
lzn.fit <- fit.variogram(lzn.vgm, model=vgm("Sph"),fit.kappa = T) # fit model
lzn.kriged <- krige(LEVEL ~ 1, s1,total_grid , model=lzn.fit)
lzn.kriged %>% as.data.frame%>%select(var1.pred)%>%cbind(all%>%
                                       filter(Season == i))%>%select(var1.pred,LEVEL)%>%View()

}                            
ven(4)
                                       #  filter(is.na(LEVEL))%>%
  ggplot(aes(x=lon, y=lat)) + geom_tile(aes(fill=var1.pred)) + coord_equal() +
  scale_fill_gradient(low = "green", high="dark red")

type1_desire = lzn.kriged$var1.pred












# the level is different scale reverse back to quantify 
mapping_scale = data.frame(
  LEVEL = c(0,1,2,3,4,5,6,7,8,9,10), 
  LEVEL_scale =c(0,5,10,20,40,80,160,320,640,1280,2560)  )

# weighted garbage adding
all = all%>%
  mutate(wplbc = 6*`Plastic bottle container`,
         wplb = 3 * `Plastic bag`,
         wdis =  2* `Disposable cup / straw / tableware`,
         wstr =  11* `Foam material`,
         wflo =  9* Float,
         wfisnr = 10 * `Fishing nets and ropes`,
         wfe = 5 * `Fishing equipment`,
         wci = 1 * `Cigarette and lighter`,
         wgj =7  * `Glass jar`,
         wm = 4 * Metal,
         wp = 0 * Paper,
         wo = 8 * Others,
  )%>%
  mutate(weighted_garbage = wplbc+wplb+wdis+wstr+wflo+wfisnr+wfe+wci+wgj+wm+wp+wo,na.rm = T)%>%
  left_join(mapping_scale)%>%
  select(-LEVEL)

# do not use the level variable for data leakage 
train = all%>%filter(!is.na(LEVEL_scale))
upload = all %>% filter(is.na(LEVEL_scale))

train%>%names()
train$LEVEL_scale
#### modeling train--> train/test --> model ####
# Full data set
train_index <- sample(1:nrow(train), nrow(train)*0.75)
data_variables <- as.matrix(train[,-52])
data_label <- train[,"LEVEL_scale"]
data_matrix <- xgb.DMatrix(data = as.matrix(train), label = unlist(data_label))
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index,]
train_matrix <- xgb.DMatrix(data = train_data, label = unlist(train_label))
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index,]
test_matrix <- xgb.DMatrix(data = test_data, label = unlist(test_label))
# full data set 
full_matrix = xgb.DMatrix(data = data_variables,label = unlist(data_label))
# upload date 
upload_data <- xgb.DMatrix(data =as.matrix(upload[-52]))

#### train and test part 1 ####
# xgboost find best cv 
{
  param <- list("objective" = "reg:linear", 
                "eval_metric" = "mae",
                'colsample_bytree' = 0.7,
                'subsample' = 0.7,
                'max_depth' = 6,
                'eta' = 0.1,
                'seed' = 70)
  set.seed(70)
  xgb_model = xgboost(param = param,  data = train_matrix, nrounds =66 , print_every_n = 100)
  train_pred = predict(xgb_model,test_matrix)
  tt = train_pred%>%
    cbind(test_label)%>%
    rename(level_scale_pred = '.')%>%
    mutate(level_scale_pred = case_when(level_scale_pred<5~0,
                                        level_scale_pred<10~1,
                                        level_scale_pred<20~2,
                                        level_scale_pred<40~3,
                                        level_scale_pred<80~4,
                                        level_scale_pred<160~5,
                                        level_scale_pred<320~6,
                                        level_scale_pred<640~7,
                                        level_scale_pred<1280~8,
                                        level_scale_pred<2560~9,
                                        TRUE~ as.numeric(10)
    ))%>%  mutate(LEVEL_scale = case_when(LEVEL_scale<5~0,
                                          LEVEL_scale<10~1,
                                          LEVEL_scale<20~2,
                                          LEVEL_scale<40~3,
                                          LEVEL_scale<80~4,
                                          LEVEL_scale<160~5,
                                          LEVEL_scale<320~6,
                                          LEVEL_scale<640~7,
                                          LEVEL_scale<1280~8,
                                          LEVEL_scale<2560~9,
                                          TRUE~ as.numeric(10)
    ))
  
  # local leaderboard
  temp=cohen.kappa(
    factor(round(tt$level_scale_pred))%>%
      cbind(
        factor(round(tt$LEVEL_scale))))
  
  temp$weighted.kappa
}
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names(train_data), model = xgb_model)
fwrite(importance_matrix,'./importance_xgb_regv3n200.csv')
#### full / upload ####
test = read_csv('./test.csv')

xgb_model = xgboost(param = param,  data = full_matrix, nrounds = 100, print_every_n = 10)
test$LEVEL= round(predict(xgb_model,upload_data))
# write
test%>%
  mutate(LEVEL = case_when(LEVEL<5~0,
                           LEVEL<10~1,
                           LEVEL<20~2,
                           LEVEL<40~3,
                           LEVEL<80~4,
                           LEVEL<160~5,
                           LEVEL<320~6,
                           LEVEL<640~7,
                           LEVEL<1280~8,
                           LEVEL<2560~9,
                           TRUE~ as.numeric(10)
  ))%>%
  mutate(ID = paste0(Station,'_',Season))%>%
  select(ID,LEVEL)%>%
  fwrite('./submission_xgboost_reg_v2_mapping_scale_n100.csv',row.names = FALSE)
