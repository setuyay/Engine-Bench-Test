setwd("G:/PHD/phdr")
#load libraries
library(caret)
library(dplyr)
library(MLmetrics)
library(dummies)
library(data.table)
library(DMwR)
library(caret)
library(tidyr)
library(ROSE)
library(RColorBrewer)
library(devtools)
library(e1071)
library(MASS)
library(car)
library(corrplot)


#####################Read data ########################

Train_AdditionalData  <- read.csv("Train_AdditionalData.csv",sep = ',',stringsAsFactors = F,header = T)
Train   <- read.csv("Train.csv",sep = ',',stringsAsFactors = F,header = T)
Test_AdditionalData  <- read.csv("Test_AdditionalData.csv",sep = ',',stringsAsFactors = F,header = T)
Test  <- read.csv("Test.csv",sep = ',',stringsAsFactors = F,header = T)
#Sample_Submission  <- read.csv("Sample_Submission.csv",sep = ',',stringsAsFactors = F,header = T)
ID<-Test$ID


################# Merge the Train and Test datasets##############

Train_AdditionalData$testApassed <- 1
Train_AdditionalData$testBpassed  <- 1
Test_AdditionalData$testApassed  <- 1
Test_AdditionalData$testBpassed  <- 1

Train = left_join(Train, Train_AdditionalData[,c("TestA", "testApassed")], by = c("ID"="TestA"))
Train = left_join(Train, Train_AdditionalData[,c("TestB", "testBpassed")], by = c("ID"="TestB"))
Test = left_join(Test, Test_AdditionalData[,c("TestA", "testApassed")], by = c("ID"="TestA"))
Test = left_join(Test, Test_AdditionalData[,c("TestB", "testBpassed")], by = c("ID"="TestB"))

Train["testApassed"][is.na(Train["testApassed"])] <- 0
Train["testBpassed"][is.na(Train["testBpassed"])] <- 0

Test["testApassed"][is.na(Test["testApassed"])] <- 0
Test["testBpassed"][is.na(Test["testBpassed"])] <- 0

Train$testApassed  <- as.factor(Train$testApassed)
Train$testBpassed  <- as.factor(Train$testBpassed)

Test$testApassed  <- as.factor(Test$testApassed)
Test$testBpassed  <- as.factor(Test$testBpassed)

########################convert all attribute to factors in train data #############

Train$y<-as.factor(Train$y)
Train$Number.of.Cylinders<-as.factor(Train$Number.of.Cylinders)
Train$material.grade<-as.factor(Train$material.grade)
Train$Lubrication<-as.factor(Train$Lubrication)
Train$Valve.Type<-as.factor(Train$Valve.Type)
Train$Bearing.Vendor<-as.factor(Train$Bearing.Vendor)
Train$Fuel.Type<-as.factor(Train$Fuel.Type)
Train$Compression.ratio<-as.factor(Train$Compression.ratio)
Train$cam.arrangement<-as.factor(Train$cam.arrangement)
Train$Cylinder.arragement<-as.factor(Train$Cylinder.arragement)
Train$Turbocharger<-as.factor(Train$Turbocharger)
Train$Varaible.Valve.Timing..VVT.<-as.factor(Train$Varaible.Valve.Timing..VVT.)
Train$Cylinder.deactivation<-as.factor(Train$Cylinder.deactivation)
Train$Direct.injection<-as.factor(Train$Direct.injection)
Train$main.bearing.type<-as.factor(Train$main.bearing.type)
Train$displacement<-as.factor(Train$displacement)
Train$piston.type<-as.factor(Train$piston.type)
Train$Max..Torque<-as.factor(Train$Max..Torque)
Train$Peak.Power<-as.factor(Train$Peak.Power)
Train$Crankshaft.Design<-as.factor(Train$Crankshaft.Design)
Train$Liner.Design.<-as.factor(Train$Liner.Design.)
Train$testApassed<-as.factor(Train$testApassed)
Train$testBpassed<-as.factor(Train$testBpassed)



########################convert all attribute to factors in test data #############


Test$Number.of.Cylinders<-as.factor(Test$Number.of.Cylinders)
Test$material.grade<-as.factor(Test$material.grade)
Test$Lubrication<-as.factor(Test$Lubrication)
Test$Valve.Type<-as.factor(Test$Valve.Type)
Test$Bearing.Vendor<-as.factor(Test$Bearing.Vendor)
Test$Fuel.Type<-as.factor(Test$Fuel.Type)
Test$Compression.ratio<-as.factor(Test$Compression.ratio)
Test$cam.arrangement<-as.factor(Test$cam.arrangement)
Test$Cylinder.arragement<-as.factor(Test$Cylinder.arragement)
Test$Turbocharger<-as.factor(Test$Turbocharger)
Test$Varaible.Valve.Timing..VVT.<-as.factor(Test$Varaible.Valve.Timing..VVT.)
Test$Cylinder.deactivation<-as.factor(Test$Cylinder.deactivation)
Test$Direct.injection<-as.factor(Test$Direct.injection)
Test$main.bearing.type<-as.factor(Test$main.bearing.type)
Test$displacement<-as.factor(Test$displacement)
Test$piston.type<-as.factor(Test$piston.type)
Test$Max..Torque<-as.factor(Test$Max..Torque)
Test$Peak.Power<-as.factor(Test$Peak.Power)
Test$Crankshaft.Design<-as.factor(Test$Crankshaft.Design)
Test$Liner.Design.<-as.factor(Test$Liner.Design.)
Test$testApassed<-as.factor(Test$testApassed)
Test$testBpassed<-as.factor(Test$testBpassed)

#####################knnImputation####

Train  <- knnImputation(Train,k=5)

Test   <- knnImputation(Test,k=5)


#### check missing value after imputation########

sum(is.na(Train))/prod(dim(Train))

#column wise NA's
colSums(is.na(Train))

unique(Train$y)

#########Checking for duplicate values#############

sum(duplicated(Train))

###############Checking for class imbalance##############

print(prop.table(table(Train$y)))
str(Train)
Train$ID <- NULL
Test$ID  <- NULL






###############Naive bayes################

########### Split the dataset into test and train ############
set.seed(1234)
trainRows=createDataPartition(Train$y,p=0.7,list = F)
train = Train[trainRows,]
validation = Train[-trainRows,]
model_nb <- naiveBayes(y~.,data = train)
train
pred_nb<-predict(model_nb,train)
pred_nb
pred_nb <- as.factor(pred_nb)
Accuracy(pred_nb,train$y)#0.79954
pred_nbva<-predict(model_nb,validation)
pred_nbva
Accuracy(pred_nbva,validation$y)#0.78252
tnb<-table(pred_nbva,validation$y)
pred_nb_test<-predict(model_nb,Test)
table(pred_nb_test)
#model (Naive Bayes)
ID<-data.frame(ID)
y<-as.data.frame(pred_nb_test)
table(y)
predictionnainv<-cbind(ID,y)
predictionnainv$y  <- predictionnainv$pred_nb_test

write.csv(predictionnainv,file ="predictionnainv.csv",row.names = F)

####################### svm  #########################
trainRows=createDataPartition(Train$y,p=0.7,list = F)
train = Train[trainRows,]
validation = Train[-trainRows,]
names(train)
object<-dummyVars(~.,data=train[,-c(1)])
predontrainn<-as.data.frame(predict(object,train[,-c(1)]))
str(predontrainn)
predonval<-as.data.frame(predict(object,validation[,-c(1)]))
predontest<-as.data.frame(predict(object,Test))
grid_radial <- expand.grid(sigma = c(0.025,0.5),C = c(1))
ctrl <- trainControl(method = "repeatedcv",search = "random", repeats = 2,classProbs = T,number = 4)


Radial.svm.tune <- train(x= predontrainn,y=train[,c(1)],
                         method = "svmRadial",
                         tuneGrid = grid_radial,
                         metric = "Accuracy", trControl = ctrl)
#on train
predssvmr<-predict(Radial.svm.tune,predontrainn)
ct2  <- table(train$y,predssvmr)
accuracy=(ct2[1,1]+ct2[2,2])/(ct2[1,1]+ct2[2,2]+ct2[1,2]+ct2[2,1])
accuracy
#on validation
predsvmr<-predict(Radial.svm.tune,predonval);predsvmr
cv2  <- table(validation$y,predsvmr)
accuracy=(cv2[1,1]+cv2[2,2])/(cv2[1,1]+cv2[2,2]+cv2[1,2]+cv2[2,1])
accuracy
#on test
svmtest<-predict(Radial.svm.tune,predontest);svmtest
table(svmtest)


#final model (SVM)
ID<-data.frame(ID)
y<-as.data.frame(svmtest)
table(y)
predictionsvm<-cbind(ID,y);predictionsvm
predictionsvm$y  <- predictionsvm$svmtest




write.csv(predictionsvm,file ="predictionsvm.csv",row.names = F)



#******logistic regression*****
model1<-glm(y~.,data=Train,family = "binomial")
summary(model1)
#predicting on validation
predictions1<-predict(model1,validation,type = "response");predictions1
library(ROCR)
rocpred<-prediction(predictions1,validation$y)
perf<-performance(rocpred,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf,colorize=TRUE,print.cutoffs.at=seq(0,1,by=0.1))
predictions=ifelse(predictions1>0.4,"pass","fail")
confusionMatrix(validation$y,predictions,positive ="pass")

#predicting on testdata
logpred<-predict(model1,Test,type="response");logpred
logpreds=ifelse(logpred>0.53,"pass","fail")
table(logpreds)

ID<-data.frame(ID)
y<-as.data.frame(logpreds)
table(y)
predictions<-cbind(ID,y);predictions
predictions$y  <- predictions$logpreds


write.csv(predictions,file ="predictionlog.csv",row.names = F)

