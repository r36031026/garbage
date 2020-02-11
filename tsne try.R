library(Rtsne)
library(tidyverse)
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
  mutate(LEVEL = LEVEL-1)
# feature engineering 
# back to train and test with all numeric columns
train = all %>% filter(!is.na(LEVEL))
sapply(train, function(x) sum(is.na(x)))
# fill NA for tsne usuage 
train = train%>%
  filter(Season==4)%>%
  mutate_if(is.numeric, ~replace_na(., -1))
train_matrix <- as.matrix(train[,1:34])
train_matrix
set.seed(42) # Set a seed if you want reproducible results
tsne_out <- Rtsne(train_matrix,perplexity = 5) # Run TSNE
plot(tsne_out$Y,col=train$LEVEL)
