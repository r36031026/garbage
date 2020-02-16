
#### featuring ####
# xgboost of multiple classification need to start from zero 
library(tidyr)
library(caret)
library(xgboost)
library(data.table)
library(dplyr)
library(psych)
library(readr)
train = read_csv('./train revise.csv')
train%>%View()
test = read_csv('./test revise.csv')
train%>%names()
temp%>%
  filter(縣市==15)%>%
  select(Location,County,縣市,Lon,Lat,Season,Seat,weighted_garbage,
#         `Plastic bottle container`,`Plastic bag`,`Disposable cup / straw / tableware`,
#         `Foam material`,Float,`Fishing nets and ropes`,`Fishing equipment`,`Cigarette and lighter`,`Glass jar`,Metal,Paper,Others,
LEVEL)%>%
#  mutate(all_garbage =  `Plastic bottle container`+`Plastic bag`+`Disposable cup / straw / tableware`+
#         `Foam material`+Float+`Fishing nets and ropes`+`Fishing equipment`+`Cigarette and lighter`+`Glass jar`+Metal+Paper+Others)%>%
  View()


a = lm(LEVEL_scale~.,data = train%>%
  select(
         `Plastic bottle container`,`Plastic bag`,`Disposable cup / straw / tableware`,
         `Foam material`,Float,`Fishing nets and ropes`,`Fishing equipment`,`Cigarette and lighter`,`Glass jar`,Metal,Paper,Others,LEVEL)%>%left_join(mapping_scale)%>%select(-LEVEL)
)
a%>%summary()
org_gar = round(a$coefficient,1)%>%data.frame()%>%
  rename(gar_coe_org = ".")%>%
  mutate(gar_coe_abs =abs(gar_coe_org ))%>%  
  mutate(garbage = row.names(data.frame(a$coefficients)))

temp = train%>%
  mutate(wplbc = org_gar$gar_coe_abs[[2]]*`Plastic bottle container`,
         wplb = org_gar$gar_coe_abs[[3]] * `Plastic bag`,
         wdis = org_gar$gar_coe_abs[[4]] * `Disposable cup / straw / tableware`,
         wstr = org_gar$gar_coe_abs[[5]] * `Foam material`,
         wflo = org_gar$gar_coe_abs[[6]] * Float,
         wfisnr = org_gar$gar_coe_abs[[7]] * `Fishing nets and ropes`,
         wfe = org_gar$gar_coe_abs[[8]] * `Fishing equipment`,
         wci = org_gar$gar_coe_abs[[9]] * `Cigarette and lighter`,
         wgj = org_gar$gar_coe_abs[[10]] * `Glass jar`,
         wm = org_gar$gar_coe_abs[[11]] * Metal,
         wp = org_gar$gar_coe_abs[[12]] * Paper,
         wo = org_gar$gar_coe_abs[[13]] * Others,
         )%>%
  mutate_all(funs(replace_na(.,0)))%>%
  mutate(weighted_garbage = wplbc+wplb+wdis+wstr+wflo+wfisnr+wfe+wci+wgj+wm+wp+wo,na.rm = T)
temp%>%View()
cor(temp$weighted_garbage,temp$LEVEL)

### build xgboost as base line ####
set.seed(707)
#### featuring ####
# xgboost of multiple classification need to start from zero 
all = train%>%
  bind_rows(test)%>%
  # separate the station part for EN and NUMERIC part for KNN 
#  mutate(Station = as.numeric(as.factor(Station)))%>%
  mutate(County = as.numeric(as.factor(County)))%>%
  mutate(Location = as.numeric(as.factor(Location)))%>%
  mutate_at(c(13:22), ~replace_na(., -1))%>%
  # make the Seat simple version 
  mutate(Seat_simple = case_when(Seat%in%c(1,2,8)~ 'north',
                                 Seat%in%c(2,3,4)~ 'west',
                                 Seat%in%c(4,5,6)~ 'south',
                                 Seat%in%c(6,7,8)~ 'east'))%>%
  mutate(Seat_simple = as.numeric(as.factor(Seat_simple)))%>%
  mutate(Season = as.factor(Season))

train = all%>%filter(!is.na(LEVEL))
test = all%>%filter(is.na(LEVEL))
test%>%glimpse()
#### modeling train--> train/test --> model ####
# Full data set
train_index <- sample(1:nrow(train), nrow(train)*0.75)
train_data = train[train_index,]
test_data = train[-train_index,]
# initial lm and step to find better variables
train_data$海岸段%>%table()
test_data$海岸段%>%table()  

lm_models = train_data %>% 
  group_by(海岸段) %>% 
  do(model = lm(LEVEL ~ Season + 縣市 + Lat+Lon+Seat + `Shore shape` + `Substrate type`, data = .))

# lm_models$model[[1]]%>%summary()
# lm_models$model[[2]]%>%summary()
# lm_models$model[[3]]%>%summary()
# lm_models$model[[4]]%>%summary()
# lm_models$model[[5]]%>%summary()
# lm_models$model[[1]][["coefficients"]]%>%as.data.frame()
# lm_models$model[[2]][["coefficients"]]%>%as.data.frame()

#lm_model = lm(formula = LEVEL ~ Season + 縣市 + Lat+Lon+Seat + `Shore shape` + `Substrate type`
#                , data = train_data ,groups='海岸段' ) 
test_data                
final = data.frame()
for( i in c(1:5)){
   final = final%>%
     bind_rows(test_data%>%
       filter(海岸段==i)%>%
        cbind( z =round(predict(lm_models$model[[i]],newdata = test_data%>%filter(海岸段==i)))))
  
  }


final%>%View()

# confusion matrix
confusionMatrix(as.factor(final$z),
                as.factor(final$LEVEL),
                mode = "everything")
# local leaderboard weighted kappa 0.66
temp=cohen.kappa(final%>%filter(海岸段==5)%>%select(z,LEVEL))
temp$weighted.kappa




# upload part 
lm_models = train%>% 
  group_by(海岸段) %>% 
  do(model = lm(LEVEL ~ Season + 縣市 + Lat+Lon+Seat + `Shore shape` + `Substrate type`, data = .))

lm_models$model[[1]]%>%summary()
lm_models$model[[2]]%>%summary()
lm_models$model[[3]]%>%summary()
lm_models$model[[4]]%>%summary()
lm_models$model[[5]]%>%summary()

#lm_model = lm(formula = LEVEL ~ Season + 縣市 + Lat+Lon+Seat + `Shore shape` + `Substrate type`
#                , data = train_data ,groups='海岸段' ) 
upload = data.frame()
for( i in c(1:5)){
  upload = upload%>%
    bind_rows(test%>%
                filter(海岸段==i)%>%
                cbind( LEVEL =round(predict(lm_models$model[[i]],newdata = test%>%filter(海岸段==i)))))
  
}





#### full / upload ####
test = read_csv('./test.csv')

upload_prediction = 
  upload%>%
  mutate(ID = paste0(Station,'_',Season))%>%
  select(ID,LEVEL)

upload_prediction%>%View()
# ready to upload format 
fwrite(upload_prediction,'./submission_lm_by_sealine.csv',row.names = FALSE)
