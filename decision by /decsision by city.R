library(caret)
library(xgboost)
library(data.table)
library(dplyr)
library(psych)
library(rpart)

train = read.csv('./train revise.csv')
test = read.csv('./test revise.csv')
### build xgboost as base line ####
train%>%View()
set.seed(707)
#### featuring ####
# xgboost of multiple classification need to start from zero 
all = train%>%
  bind_rows(test)%>%
  # separate the station part for EN and NUMERIC part for KNN 
  #  mutate(station_num = gsub("[^0-9.-]","",Station),
  #         station_num = as.numeric(station_num))%>%
  #  mutate(station_EN = gsub("[^A-Z]","",Station),
  #         station_EN = as.numeric(station_EN))%>%
  mutate(Station = as.numeric(as.factor(Station)))%>%
  mutate(County = as.numeric(as.factor(County)))%>%
  mutate(Location = as.numeric(as.factor(Location)))%>%
  # make the Seat simple version 
  mutate(Seat_simple = case_when(Seat%in%c(1,2,8)~ 'north',
                                 Seat%in%c(2,3,4)~ 'west',
                                 Seat%in%c(4,5,6)~ 'south',
                                 Seat%in%c(6,7,8)~ 'east'))%>%
  # Seat_simple and Season 
  mutate(Seat_and_season  = case_when(Seat_simple == 'north'& Season %in%c(4)~ 2,
                                      Seat_simple == 'north' & Season %in% c(3)~1,
                                      Seat_simple == 'south' & Season %in% c(1)~1,
                                      Seat_simple == 'south' & Season %in% c(2)~2,
                                      Seat_simple == 'west' & Season %in% c(1)~1,
                                      Seat_simple == 'west' & Season %in% c(2)~2,
                                      TRUE~as.numeric(0)
  ))%>%
  # kill the factor season 
  mutate(Seat_simple = as.numeric(as.factor(Seat_simple)))%>%
  mutate(wplbc = 6* Plastic.bottle.container,
         wplb = 3 * Plastic.bag,
         wdis =  2* Disposable.cup...straw...tableware,
         wstr =  11* Foam.material,
         wflo =  9* Float,
         wfisnr = 10 * Fishing.nets.and.ropes,
         wfe = 5 * Fishing.equipment,
         wci = 1 * Cigarette.and.lighter,
         wgj =7  * Glass.jar,
         wm = 4 * Metal,
         wp = 0 * Paper,
         wo = 8 * Others,
  )%>%
  # top 3 appear and top 3 biggest  
  mutate(top_3_appear = Foam.material+Fishing.nets.and.ropes+Plastic.bottle.container)%>%
  mutate(top_3_biggest = Fishing.nets.and.ropes+ Float +Others ) %>%
  group_by(Season,Seat)%>%
  mutate(ss_Foam_mean = mean(Foam.material,na.rm=T),
         ss_Fishingnr_mean = mean(Fishing.nets.and.ropes),
         ss_Plasticbc_mean = mean(Plastic.bottle.container))%>%
  ungroup()

train_final = all%>%filter(!is.na(LEVEL))
upload_final = all%>%filter(is.na(LEVEL))

upload_final%>%View()
decision_flow<-function(i){
  tmp = upload_final%>%
    filter(row_number()==i)
  if(tmp$Location %in%(train_final$Location)){
    decision_model  = rpart(formula = LEVEL~.,data = train_final%>%filter(Location==tmp$Location))
    tmp$LEVEL = predict(decision_model, newdata=tmp)
    cat('location')
  }else if(tmp$County%in%(train_final$County)){
    decision_model  = rpart(formula = LEVEL~.,data = train_final%>%filter(County==tmp$County))
    tmp$LEVEL = predict(decision_model, newdata=tmp)
    
    cat('county')  
  }else if(tmp$縣市%in%(train_final$縣市)){
    decision_model  = rpart(formula = LEVEL~.,data = train_final%>%filter(縣市==tmp$縣市))
    tmp$LEVEL = predict(decision_model, newdata=tmp)
    
    cat('city')  
  }
  return(tmp)
}
decision_flow(4)
# cart
z =lapply(c(1:(upload_final%>%nrow())),decision_flow)%>%bind_rows()
test= read.csv('./test revise.csv')

test$LEVEL = z$LEVEL
test%>%
  mutate(LEVEL = round(LEVEL))%>%
  mutate(ID = paste0(Station,'_',Season))%>%
  select(ID,LEVEL)%>%
  fwrite('./decision by /decision_flow_v1.csv',row.names = FALSE)
