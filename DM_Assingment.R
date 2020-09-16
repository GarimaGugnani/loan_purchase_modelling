getwd()
setwd("G:/Project 3-Data Mining")
loan_data=read.csv("Thera Bank_Personal_Loan_Modelling.csv", sep=",", header=TRUE, na.strings=c(" ", "NA"))
#view datasheet
View(loan_data)
#find class of each variable
str(loan_data)
loan_data$Personal.Loan=as.factor(loan_data$Personal.Loan)#convert target variable into factor variable
loan_data$Securities.Account=as.factor(loan_data$Securities.Account)
loan_data$CD.Account=as.factor(loan_data$CD.Account)
loan_data$Online=as.factor(loan_data$Online)
loan_data$CreditCard=as.factor(loan_data$CreditCard)
loan_data$Education=as.factor(loan_data$Education)
loan_data=loan_data[,-c(1,5)]#remove id and zip code variable
#summary of each variable
summary(loan_data)#18 NAs in family members #negative experience values
#check for NAs
any(is.na(loan_data))
colSums(is.na(loan_data))
loan_data=na.omit(loan_data)
any(is.na(loan_data))
loan_data$Experience..in.years.=abs(loan_data$Experience..in.years.)
summary(loan_data)
View(loan_data)


boxplot(loan_data$Age..in.years.)
boxplot(loan_data$Experience..in.years.)
boxplot(loan_data$Income..in.K.month.)
boxplot(loan_data$Family.members)
boxplot(loan_data$CCAvg)
boxplot(loan_data$Mortgage)
plot(loan_data[,c(1,2,3,4,5,6,7,8)])
library(ggplot2)
ggplot(loan_data, aes(x= Experience..in.years. , y= Age..in.years.)) + geom_jitter(width = 0.1)



ggplot(loan_data, aes(x= Education ,y= Income..in.K.month.,color = Personal.Loan)) + geom_jitter(width = 0.1, alpha = 0.5)
ggplot(loan_data, aes(x= Family.members ,y= Income..in.K.month.,color = Personal.Loan)) + geom_jitter(width = 0.1, alpha = 0.5)


ggplot(loan_data, aes(x = Income..in.K.month., fill = Personal.Loan)) + 
  geom_bar() + 
  labs(x = "Income", y = "Loan count", fill = "Personal Loan")
#securities, family, CD account, online , credit card
ggplot(loan_data, aes(x = Securities.Account, fill = Personal.Loan)) + 
  geom_bar() + 
  labs(x = "Securities Account", y = "Loan count", fill = "Personal Loan")

ggplot(loan_data, aes(x = Online, fill = Personal.Loan)) + 
  geom_bar() + 
  labs(x = "Online", y = "Loan count", fill = "Personal Loan")

ggplot(loan_data, aes(x = CreditCard, fill = Personal.Loan)) + 
  geom_bar() + 
  labs(x = "CreditCard", y = "Loan count", fill = "Personal Loan")

ggplot(loan_data, aes(x = CD.Account, fill = Personal.Loan)) + 
  geom_bar() + 
  labs(x = "CD Account", y = "Loan count", fill = "Personal Loan")



set.seed(1000)
trainindex=sample(1:nrow(loan_data), 0.70*nrow(loan_data))
train=loan_data[trainindex,]
test=loan_data[-trainindex,]
#splitting data
library(caTools)
split <- sample.split(loan_data$Personal.Loan, SplitRatio = 0.7)
#we are splitting the data such that we have 70% of the data is Train Data and 30% of the data is my Test Data

train<- subset(loan_data, split == TRUE)
test<- subset( loan_data, split == FALSE)
nrow(train)
sum(train$Personal.Loan=="1")/nrow(train)
sum(test$Personal.Loan=="1")/nrow(test)

library(rpart)
library(rpart.plot)
tree=rpart(formula = Personal.Loan~., data=train, method="class", minbucket=2, cp=0)
print(tree)
rpart.plot(tree)
printcp(tree)
plotcp(tree)
ptree=prune(tree, cp=0.006, "CP")
print(ptree)
rpart.plot(ptree)

train$predict=predict(ptree, data=train, type="class")
train$prob1=predict(ptree, data=train, type="prob")[,"1"]
View(train)
test$predict=predict(ptree, test, type="class")
test$prob1=predict(ptree, test, type="prob")[,"1"]

library(randomForest)
set.seed(1000)
Rforest=randomForest(Personal.Loan~., data=loan_data, ntree=501, mtry=5, nodesize=10, importance=TRUE)
print(Rforest)
Rforest$err.rate
plot(Rforest)
importance(Rforest)
set.seed(1000)
tRfprest=tuneRF(x=train[,-c(8)], y=train$Personal.Loan, mtryStart = 5, stepFactor = 1.5, ntreeTry = 501, improve = 0.0001, nodesize=10, trace=TRUE, plot= TRUE, doBest = TRUE, importance=TRUE)

Rforest.Final=randomForest(Personal.Loan~., data=loan_data, ntree=501, mtry=5, nodesize=10, importance=TRUE)
plot(Rforest.Final)

train$predictRF=predict(Rforest.Final, train, type="class")
train$prob1RF=predict(Rforest.Final, train, type="prob")[,"1"]
View(train)
test$predictRF=predict(Rforest.Final, test, type="class")
test$prob1RF=predict(Rforest.Final, test, type="prob")[,"1"]
#for CART
table(train$Personal.Loan, train$predict)
Accuracy=(3137+313)/3488
Accuracy
Sensitivity=313/(313+22)
Sensitivity
Specificity=3137/(3137+16)
Specificity


table(test$Personal.Loan, test$predict)
Accuracy=(1342+128)/1494
Accuracy
Sensitivity=128/(128+15)
Sensitivity
Specificity=1342/(1342+9)
Specificity

library(ROCR)
library(ineq)
library(InformationValue)


preobjtrain=prediction(train$prob1,train$Personal.Loan)
preftrain=performance(preobjtrain,"tpr","fpr")
plot(preftrain)#ROC curve

preobjtest=prediction(test$prob1,test$Personal.Loan)
preftest=performance(preobjtest,"tpr","fpr")
plot(preftest)

auctrain=performance(preobjtrain,"auc")
auctrain=as.numeric(auctrain@y.values)
auctrain

auctest=performance(preobjtest,"auc")
auctest=as.numeric(auctest@y.values)
auctest

KStrain=max(preftrain@y.values[[1]]-preftrain@x.values[[1]])
KStrain

KStest=max(preftest@y.values[[1]]-preftest@x.values[[1]])
KStest

Ginitrain=ineq(train$prob1, "gini")
Ginitrain

Ginitest=ineq(test$prob1, "gini")
Ginitest

#for random forest
table(test$Personal.Loan, test$predictRF)
Accuracy=(1350+132)/1494
Accuracy
Sensitivity=132/(143)
Sensitivity
Specificity=1350/(1351)
Specificity


table(train$Personal.Loan, train$predictRF)
Accuracy=(3131+321)/3488
Accuracy
Sensitivity=321/(321+14)
Sensitivity
Specificity=3151/(3153)
Specificity

library(ROCR)
library(ineq)
library(InformationValue)


preobjtrainRF=prediction(train$prob1RF,train$Personal.Loan)
preftrainRF=performance(preobjtrain,"tpr","fpr")
plot(preftrain)#ROC curve

preobjtestRF=prediction(test$prob1RF,test$Personal.Loan)
preftestRF=performance(preobjtestRF,"tpr","fpr")
plot(preftestRF)

auctrainRF=performance(preobjtrainRF,"auc")
auctrainRF=as.numeric(auctrainRF@y.values)
auctrainRF

auctestRF=performance(preobjtestRF,"auc")
auctestRF=as.numeric(auctestRF@y.values)
auctestRF

KStrainRF=max(preftrainRF@y.values[[1]]-preftrainRF@x.values[[1]])
KStrainRF

KStestRF=max(preftestRF@y.values[[1]]-preftestRF@x.values[[1]])
KStestRF

GinitrainRF=ineq(train$prob1RF, "gini")
GinitrainRF

GinitestRF=ineq(test$prob1RF, "gini")
GinitestRF

