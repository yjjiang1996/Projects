set.seed(423)
train_1_total=read.csv('hospital_train_data1.csv')
train_2_total=read.csv('hospital_train_data2.csv')
fake_test_1=read.csv('hospital_test_data1.csv')
fake_test_2=read.csv('hospital_test_data2.csv')

cols_train_1=c(2,3,5,6,7,14,15,16,17,18)
train_1_total[cols_train_1]=lapply(train_1_total[cols_train_1], factor)
train_1_total[cols_train_1]=lapply(train_1_total[cols_train_1], as.numeric)
train_1_total[cols_train_1]=lapply(train_1_total[cols_train_1], factor)
train_1_total[-cols_train_1]=lapply(train_1_total[-cols_train_1], as.numeric)
train_1_total$RETURN=as.factor(train_1_total$RETURN)

train_1_obs=nrow(train_1_total)
valid_obs = sample(train_1_obs, 0.2*train_1_obs)
valid_1<- train_1_total[valid_obs,]
train_1<- train_1_total[-valid_obs,]

cols_fake_test_1=c(2,3,5,6,7,14,15,16,17,18)
fake_test_1[cols_fake_test_1]=lapply(fake_test_1[cols_fake_test_1],factor)
fake_test_1[cols_fake_test_1]=lapply(fake_test_1[cols_fake_test_1],as.numeric)
fake_test_1[cols_fake_test_1]=lapply(fake_test_1[cols_fake_test_1],factor)
fake_test_1[-cols_fake_test_1]=lapply(fake_test_1[-cols_fake_test_1],as.numeric)
fake_test_1$RETURN=as.factor(fake_test_1$RETURN)
baseline_fake_test_1=1-mean(as.numeric(fake_test_1$RETURN)-1)

cols_train_2=c(2,3,5,6,7,14,15,16,17,18,19,20,21,22,24,25)
train_2_total[cols_train_2]=lapply(train_2_total[cols_train_2], factor)
train_2_total[cols_train_2]=lapply(train_2_total[cols_train_2], as.numeric)
train_2_total[cols_train_2]=lapply(train_2_total[cols_train_2], factor)
train_2_total[-cols_train_2]=lapply(train_2_total[-cols_train_2], as.numeric)
train_2_total$RETURN=as.factor(train_2_total$RETURN)

train_2_obs=nrow(train_2_total)
valid_obs = sample(train_2_obs, 0.2*train_2_obs)
valid_2<- train_2_total[valid_obs,]
train_2<- train_2_total[-valid_obs,]

cols_fake_test_2=c(2,3,5,6,7,14,15,16,17,18,19,20,21,22,24,25)
fake_test_2[cols_fake_test_2]=lapply(fake_test_2[cols_fake_test_2],factor)
fake_test_2[cols_fake_test_2]=lapply(fake_test_2[cols_fake_test_2],as.numeric)
fake_test_2[cols_fake_test_2]=lapply(fake_test_2[cols_fake_test_2],factor)
fake_test_2[-cols_fake_test_2]=lapply(fake_test_2[-cols_fake_test_2],as.numeric)
fake_test_2$RETURN=as.factor(fake_test_2$RETURN)
baseline_fake_test_2=1-mean(as.numeric(fake_test_2$RETURN)-1)

########
train_1_X=as.matrix(model.matrix(~.-1,train_1_total[,-c(1,21)]))
train_1_y=as.numeric(train_1_total$RETURN)-1
train_2_X=as.matrix(model.matrix(~.-1,train_2_total[,-c(1,27)]))
train_2_y=as.numeric(train_2_total$RETURN)-1

library(xgboost)
dtrain_1=xgb.DMatrix(train_1_X,label=train_1_y)
dtrain_2=xgb.DMatrix(train_2_X,label=train_2_y)

#best_param = list()
best_error = Inf
best_error_index = 0

for (iter in 1:500) {
  param <- list(objective = 'binary:logistic',
                eval_metric = "error",
                num_class = 1,
                max_depth = sample(1:20, 1),
                eta = runif(1, .01, 1),
                gamma = runif(1, 0.001, 10), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 500
  cv.nfold = 5
  mdcv <- xgb.cv(data=dtrain_1, params = param, nthread=3, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stopping_rounds =8, maximize=FALSE)
  
  min_error_index = mdcv$best_iteration
  min_error = mdcv$evaluation_log[min_error_index]$test_error_mean
  
  if (min_error < best_error) {
    best_error = min_error
    best_error_index = min_error_index
    best_param = param
  }
}

nround = best_error_index

library(xgboost)
xgb_cv_1 <- xgb.train(data=dtrain_1, params=best_param, nrounds=nround, nthread=6)
xgb_cv_1_predict=predict(xgb_cv_1,newdata = xgb.DMatrix(as.matrix(model.matrix(~.-1,fake_test_1[,-c(1,21)])),label=as.numeric(fake_test_1$RETURN)-1))
xgb_cv_1_class=ifelse(xgb_cv_1_predict>0.5,1,0)
xgb_cv_1_table=table(fake_test_1$RETURN,xgb_cv_1_class)
(xgb_cv_1_table[1,1]+xgb_cv_1_table[2,2])/sum(xgb_cv_1_table)
xgb_cv_1_table

TP=xgb_cv_1_table[2,2]
TN=xgb_cv_1_table[1,1]
FN=xgb_cv_1_table[2,1]
FP=xgb_cv_1_table[1,2]

RECALL=TP/(TP+FN)
RECALL
PRECISION=TP/(TP+FP)
PRECISION
F1_score=2*TP/(2*TP+FN+FP)
F1_score

##
bset_param_2=list()
best_error_2 = Inf
best_error_2_index = 0

for (iter in 1:500) {
  param <- list(objective = 'binary:logistic',
                eval_metric = "error",
                num_class = 1,
                max_depth = sample(1:20, 1),
                eta = runif(1, .01, 1),
                gamma = runif(1, 0.001, 10), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 500
  cv.nfold = 5
  mdcv <- xgb.cv(data=dtrain_2, params = param, nthread=3, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stopping_rounds =8, maximize=FALSE)
  
  min_error_index = mdcv$best_iteration
  min_error = mdcv$evaluation_log[min_error_index]$test_error_mean
  
  if (min_error < best_error) {
    best_error_2 = min_error
    best_error_2_index = min_error_index
    best_param_2 = param
  }
}

nround_2 = best_error_index

library(xgboost)
xgb_cv_2 <- xgb.train(data=dtrain_2, params=best_param_2, nrounds=nround_2, nthread=6)
xgb_cv_2_predict=predict(xgb_cv_2,newdata = xgb.DMatrix(as.matrix(model.matrix(~.-1,fake_test_2[,-c(1,27)])),label=as.numeric(fake_test_2$RETURN)-1))
xgb_cv_2_class=ifelse(xgb_cv_2_predict>0.5,1,0)
xgb_cv_2_table=table(fake_test_2$RETURN,xgb_cv_2_class)
(xgb_cv_2_table[1,1]+xgb_cv_2_table[2,2])/sum(xgb_cv_2_table)
xgb_cv_2_table

TP=xgb_cv_2_table[2,2]
TN=xgb_cv_2_table[1,1]
FN=xgb_cv_2_table[2,1]
FP=xgb_cv_2_table[1,2]

RECALL=TP/(TP+FN)
RECALL
PRECISION=TP/(TP+FP)
PRECISION
F1_score=2*TP/(2*TP+FN+FP)
F1_score



######predict
test_1=read.csv('hospital_testing_data1.csv')
test_2=read.csv('hospital_testing_data2.csv')

library(tibble)
test_1=add_column(test_1,'RETURN'=rep(0,nrow(test_1)), .after = 20)

cols_fake_test_1=c(2,3,5,6,7,14,15,16,17,18)
test_1[cols_fake_test_1]=lapply(test_1[cols_fake_test_1],factor)
test_1[cols_fake_test_1]=lapply(test_1[cols_fake_test_1],as.numeric)
test_1[cols_fake_test_1]=lapply(test_1[cols_fake_test_1],factor)
test_1[-cols_fake_test_1]=lapply(test_1[-cols_fake_test_1],as.numeric)
test_1$RETURN=as.factor(test_1$RETURN)

xgb_test_1_predict=predict(xgb_cv_1,newdata = xgb.DMatrix(as.matrix(model.matrix(~.-1,test_1[,-c(1,21)])),label=test_1$RETURN))
xgb_test_1_class=ifelse(xgb_test_1_predict>0.5,'Yes','No')
predict_1=data.frame('INDEX'=test_1$INDEX,'RETURN'=xgb_test_1_class)
write.csv(predict_1,'predict_1.csv')

baseline_fake_test_1
baseline_fake_test_2


#####
library(klaR)
bayes=NaiveBayes(as.factor(RETURN)~.-INDEX,data=train_1_total)
bayes_predict=predict(bayes,newdata = fake_test_1)
bayes_class=ifelse(bayes_predict$posterior[,2]>0.7,1,0)
bayes_table=table(fake_test_1$RETURN,bayes_class)
bayes_table
(bayes_table[1,1]+bayes_table[2,2])/sum(bayes_table)

TP=bayes_table[2,2]
TN=bayes_table[1,1]
FN=bayes_table[2,1]
FP=bayes_table[1,2]

RECALL=TP/(TP+FN)
RECALL
PRECISION=TP/(TP+FP)
PRECISION
F1_score=2*TP/(2*TP+FN+FP)
F1_score

##
bayes_2=NaiveBayes(as.factor(RETURN)~.-INDEX,data=train_2_total[,-c(28,33)])
bayes_2_predict=predict(bayes_2,newdata = fake_test_2[,-c(28,33)])
bayes_2_class=ifelse(bayes_2_predict$posterior[,2]>1,1,0)
bayes_2_table=table(fake_test_2$RETURN,bayes_2_class)
bayes_2_table
(bayes_2_table[1,1]+bayes_2_table[2,2])/sum(bayes_2_table)

TP=bayes_2_table[2,2]
TN=bayes_2_table[1,1]
FN=bayes_2_table[2,1]
FP=bayes_2_table[1,2]

RECALL=TP/(TP+FN)
RECALL
PRECISION=TP/(TP+FP)
PRECISION
F1_score=2*TP/(2*TP+FN+FP)
F1_score

##
log_1=glm(RETURN~.-INDEX,data=train_1_total,family = 'binomial')
log_1_predict=predict(log_1,newdata=fake_test_1,type='response')
log_1_class=ifelse(log_1_predict>0.52,1,0)
log_1_table=table(fake_test_1$RETURN,log_1_class)
(log_1_table[1,1]+log_1_table[2,2])/sum(log_1_table)
log_1_table
TP=log_1_table[2,2]
TN=log_1_table[1,1]
FN=log_1_table[2,1]
FP=log_1_table[1,2]

RECALL=TP/(TP+FN)
RECALL
PRECISION=TP/(TP+FP)
PRECISION
F1_score=2*TP/(2*TP+FN+FP)
F1_score


#####
library(randomForest)

rf_1=randomForest(RETURN~.-INDEX,data=train_1_total,ntree=500,mtry=5,importance=TRUE)
rf_1_predict=predict(rf_1,newdata=fake_test_1)
rf_1_table=table(fake_test_1$RETURN,rf_1_predict)
rf_1_table
(rf_1_table[1,1]+rf_1_table[2,2])/sum(rf_1_table)
