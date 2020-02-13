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
  mutate_at(c(13:22), ~replace_na(., -1))
# feature engineering 
# back to train and test with all numeric columns
train = all %>% filter(!is.na(LEVEL))
upload = all %>% filter(is.na(LEVEL))



#### modeling train--> train/test --> model ####
# Full data set
train_index <- sample(1:nrow(train), nrow(train)*0.75)
train_data = train[train_index,]
test_data = train[-train_index,]
# initial lm and step to find better variables
lm_model= lm(LEVEL~.,data = train_data)
step(lm_model,direction = 'both')

lm_model = lm(formula = LEVEL ~ Station + County + Lat + 縣市 + 海岸段 + 
    Seat + X1暴露岩岸 + X2暴露人造結構物 + X5砂礫混合灘 + 
    X8遮蔽岩岸 + Plastic.bottle.container + Plastic.bag + 
    Foam.material + Float + Fishing.nets.and.ropes + Metal + 
    Others, data = train_data)
lm_model%>%summary# R-square from 40.072 to 0.4372
test_data$pred = predict(lm_model,newdata = test_data)
test_data = test_data%>%
  mutate(pred = round(pred))

# confusion matrix
confusionMatrix(factor(test_data$pred),
                factor(test_data$LEVEL),
                mode = "everything")
# local leaderboard weighted kappa 0.66
temp=cohen.kappa(factor(test_data$pred)%>%cbind(
  factor(test_data$LEVEL)))
temp$weighted.kappa


#### lm upload part training ####
lm_model= lm(LEVEL~.,data = train)
step(lm_model,direction = 'both')
lm_model = lm(formula = LEVEL ~ Station + Season + Lon + 縣市 + 海岸段 + 
     Seat + Substrate.type + X1暴露岩岸 + X5砂礫混合灘 + 
     X8遮蔽岩岸 + Plastic.bottle.container + Foam.material + 
     Float + Fishing.nets.and.ropes + Cigarette.and.lighter + 
     Metal + Others, data = train)
a = summary(lm_model)['coefficients'] # R-square 0.4814
b=a$coefficients%>%as.data.frame()
b$Estimate
upload$LEVEL = predict(lm_model,newdata = upload)
#### full / upload ####
test = read.csv('./test.csv')

upload_prediction = 
  upload%>%
  mutate(LEVEL = round(LEVEL))%>%
  select(LEVEL)%>% 
  cbind(test)%>%
  mutate(ID = paste0(Station,'_',Season))%>%
  select(ID,LEVEL)
upload_prediction%>%View()

# ready to upload format 
fwrite(upload_prediction,'./submission_lm_v1.csv',row.names = FALSE)
