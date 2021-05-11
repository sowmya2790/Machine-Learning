#mkt ptoject
#invistico airline survey (https://www.kaggle.com/sjleshrac/airlines-customer-satisfaction)

#data import
data=read.csv("C:/Users/Pzhu7/Desktop/MKT591/Invistico_Airline.csv",header=TRUE)
dim(data)
head(data)

#library
library(ggplot2)
library(scales)
library(plyr)
library(cluster)
library(MASS)

###############################
#####Descriptive analysis######
###############################

#barplots
ggplot(data, aes(x=Age, y=..count..))+geom_bar(position="dodge")
ggplot(data, aes(x=Gender, y=..count..))+geom_bar(position="dodge")
ggplot(data, aes(x=satisfaction, y=..count..))+geom_bar(position="dodge")
ggplot(data, aes(x=Class, y=..count..))+geom_bar(position="dodge")
ggplot(data, aes(x=Customer.Type, y=..count..))+geom_bar(position="dodge")
ggplot(data, aes(x=Type.of.Travel, y=..count..))+geom_bar(position="dodge")

#percentage_view_plots
ggplot(data, aes(x=Customer.Type, y=..count../sum(..count..)))+geom_bar(position = "dodge")

ggplot(data, aes(x=Gender, y=..count../sum(..count..)))+geom_bar(position="dodge")

ggplot(data, aes(x=satisfaction, y=..count../sum(..count..)))+geom_bar(position="dodge")

ggplot(data, aes(x=Class, y=..count../sum(..count..)))+geom_bar(position="dodge")

ggplot(data, aes(x=Type.of.Travel, y=..count../sum(..count..)))+geom_bar(position="dodge")

#boxplots
ggplot(data, aes(x = Customer.Type, y= Age, fill= satisfaction)) + geom_boxplot()
###Insight: larger range of age in loyal passengers reported "dissatisfied"

ggplot(data, aes(x = Type.of.Travel, y = Age, fill= satisfaction)) + geom_boxplot()
##reverse boxplot1

ggplot(data, aes(x = Class, y = Age,fill =satisfaction)) + geom_boxplot()


#density
ggplot(data, aes(Flight.Distance,fill=factor(satisfaction)))+  geom_density(alpha = 0.5)
#highest density point for satisfied; dissatisfied

ggplot(data, aes(Age,fill=factor(satisfaction)))+  geom_density(alpha = 0.5)
#highest dissatisfied point falls on 20s
#highest satisfied point falls on 40s
#sample service column density check
ggplot(data, aes(Seat.comfort ,fill=factor(satisfaction)))+  geom_density(alpha = 0.5)
ggplot(data, aes(Leg.room.service,fill=factor(satisfaction)))+  geom_density(alpha = 0.5)
ggplot(data, aes(Inflight.entertainment ,fill=factor(satisfaction)))+  geom_density(alpha = 0.5)

#crosstable
library(gmodels)
cross=data
head(cross)
cross$satisfaction=ifelse(cross$satisfaction=="satisfied",1,0)
cross$Customer.Type=ifelse(cross$Customer.Type=="Loyal Customer",1,0)
crosstable= CrossTable(cross$satisfaction,cross$Customer.Type)
summary(crosstable)
source("http://pcwww.liv.ac.uk/~william/R/crosstab.r")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Seat.comfort"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Departure.Arrival.time.convenient"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Food.and.drink"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Gate.location"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Inflight.wifi.service"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Inflight.entertainment"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Online.support"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Ease.of.Online.booking"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("On.board.service"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Leg.room.service"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Baggage.handling"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Checkin.service"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Cleanliness"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Online.boarding"), type = "c")
#crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Departure.Delay.in.Minutes"), type = "c")
#crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Arrival.Delay.in.Minutes"), type = "c")
##################################################
################Train-Test-Split##################
##################################################
data=read.csv("C:/Users/Pzhu7/Desktop/MKT591/Invistico_Airline.csv",header=TRUE)
dim(data)
head(data)
#Assign column tags
data$satisfaction=as.factor(data$satisfaction)
data$Gender=as.factor(data$Gender)
data$Customer.Type=as.factor(data$Customer.Type)
data$Type.of.Travel=as.factor(data$Type.of.Travel)
data$Class=as.factor(data$Class)
data$Flight.Distance=scale(data$Flight.Distance)
data$Departure.Delay.in.Minutes=scale(data[,22])#departure delay
data$Arrival.Delay.in.Minutes=scale(data[,23])#arrival delay

head(data)
dim(data)
set.seed(12345)
train_split=sample(seq_len(nrow(data)), size=0.7*nrow(data))


train=data[train_split,]#70%
dim(train)
head(train)
test=data[-train_split,]#30%
dim(test)
head(test)

#write_csv
#write.csv(train,'train_project3.csv')
#write.csv(test,'test_project3.csv')

################################################
######segmentation - clustering Analysis########
################################################
library(cluster)
#read train_project.csv
train=read.csv("C:/Users/Pzhu7/Desktop/MKT591/train_project3.csv",header=TRUE)
head(train)
dim(train)

cluster_train = train[,2:23]# remove target column
#cluster_train = cluster_train[,1:6]#remove survey columns
#cluster_train=cluster_train[,1:20]#remove last 2 delay in min columns
head(cluster_train)
dim(cluster_train)

#transfer categorical to as.factor
cluster_train$Gender=ifelse(cluster_train$Gender=="Male",0,1)
cluster_train$Customer.Type=ifelse(cluster_train$Customer.Type=="Loyal Customer",0,1)
cluster_train$Type.of.Travel=ifelse(cluster_train$Type.of.Travel=="Business travel",0,1)
cluster_train$Class=ifelse(cluster_train$Class=="Business",0,ifelse(cluster_train$Class=="Eco",0.5,1))

#summ function from txt
seg.summ<-function(data,groups){
  aggregate(data, list(groups), function(x) mean(as.numeric(x)))
} # computing group-level mean values

#kmeans-method
set.seed(1000)
kmeans=kmeans(cluster_train, centers=3)#centers=# can change
kmeans$cluster


summary=seg.summ(cluster_train, kmeans$cluster)#computing segment-level means
summary
write.csv(summary,'seg.summ+k=3.csv')

library(fpc)
plotcluster(cluster_train, kmeans$cluster)
#plot kmeans cluster (take some time for plotting)
#data visualization for kmeans (better understanding the structure of datasets
#identify subgroups of passengers


################################################
######satisfaction-classification Analysis######
################################################

#read train_project.csv
train=read.csv("C:/Users/Pzhu7/Desktop/MKT591/train_project3.csv",header=TRUE)
head(train)
dim(train)
test=read.csv("C:/Users/Pzhu7/Desktop/MKT591/test_project3.csv",header=TRUE)
test_target=as.factor(test[,1])
test1=test[,-1]#remove target column
dim(test)
head(test1)
dim(test1)
head(test_target)

#Seat.comfort+Departure.Arrival.time.convenient+Food.and.drink+Gate.location+Inflight.wifi.service+Inflight.entertainment+Online.support+
#Ease.of.Online.booking+ On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Cleanliness+Online.boarding
#+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes

head(test1)
##############full logistic regression
reg1=glm(factor(satisfaction)~., family=binomial,data=train)
summary(reg1)
BIC(reg1)#70223.32

pred=predict.glm(reg1,newdata=test1,type='response')
ctv=table(test_target, pred)
ctv
pred.table=ifelse(pred>0.5,1,0)#crosstable view
table=table(test_target, pred.table)
table
(table[2,2]+table[1,1])/dim(test)[1]#83.4%
#pred.table
#test_target        0     1
#dissatisfied 14272  3235
#satisfied     3220 18237

##############logistic regression with only survey columns
reg2=glm(factor(satisfaction)~Seat.comfort+Departure.Arrival.time.convenient+Food.and.drink+Gate.location+Inflight.wifi.service+Inflight.entertainment+Online.support+Ease.of.Online.booking+ On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Cleanliness+Online.boarding+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes, 
         family=binomial,data=train)
summary(reg2)####include insignificant ivs
BIC(reg2)#80101.86 reg2>reg1
reg3=glm(factor(satisfaction)~ Seat.comfort+Departure.Arrival.time.convenient+Food.and.drink+Gate.location+Inflight.wifi.service+Inflight.entertainment+Online.support+Ease.of.Online.booking+ On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes, 
         family=binomial,data=train)
summary(reg3)####cleaness, online boarding removed
BIC(reg3)#80080.73 reg2>reg3>reg1

pred2=predict.glm(reg3,newdata=test1,type='response')
summary(pred2)

ctv2=table(test_target, pred2)
ctv2

pred2.table=ifelse(pred2>0.5,1,0)#crosstable view
table=table(test_target, pred2.table)
table
(table[2,2]+table[1,1])/dim(test)[1]# hit rate #0.8064624 #increased hit rate

#pred2.table
#test_target        0     1
#dissatisfied 13635  3872
#satisfied     3669 17788

##############logistic regression without C_Type
reg4=glm(factor(satisfaction)~ Gender+Age+Type.of.Travel+Class+Flight.Distance+Seat.comfort+Departure.Arrival.time.convenient+Food.and.drink+Gate.location+Inflight.wifi.service+Inflight.entertainment+Online.support+Ease.of.Online.booking+ On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Cleanliness+Online.boarding+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes, 
         family=binomial,data=train)
summary(reg4)###'include insignificant ivs
BIC(reg4)#74940.81
reg5=glm(factor(satisfaction)~ Gender+Age+Type.of.Travel+Class+Flight.Distance+Seat.comfort+Departure.Arrival.time.convenient+Food.and.drink+Gate.location+Inflight.wifi.service+Inflight.entertainment+Online.support+Ease.of.Online.booking+ On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Online.boarding+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes, 
         family=binomial,data=train)
summary(reg5)###cleanliness removed
BIC(reg5)#74929.43

pred3=predict.glm(reg5,newdata=test1,type='response')
summary(pred3)

ctv3=table(test_target, pred3)
ctv3

pred3.table=ifelse(pred3>0.5,1,0)#crosstable view
table2=table(test_target, pred3.table)
table2
(table2[2,2]+table2[1,1])/dim(test)[1]# hit rate #0.8222975

#pred3.table
#test_target        0     1
#dissatisfied 13972  3535
#satisfied     3389 18068

###############randomForest prediction 1 
library(randomForest)
set.seed(1000)
attach(train)
rf1=randomForest(factor(satisfaction)~ 
                   Gender+Age+Type.of.Travel+Class+Flight.Distance+Seat.comfort+Departure.Arrival.time.convenient+Food.and.drink+Gate.location+Inflight.wifi.service+Inflight.entertainment+Online.support+Ease.of.Online.booking+ On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Cleanliness+Online.boarding+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes, ntree=50)
predict1=predict(rf1, test1, predict.all=TRUE)
head(test1)
summary(predict1)

predict1$aggregate
cross1=table(test_target, predict1$aggregate)
cross1
(cross1[2,2]+cross1[1,1])/dim(test)[1]# hit rate #0.951314

#test_target    dissatisfied satisfied
#dissatisfied        16735       772
#satisfied            1125     20332

##############randomForest prediction 2 
set.seed(1000)
rf2=randomForest(factor(satisfaction)~ 
                   Gender+Age+Type.of.Travel+Class+Flight.Distance+Seat.comfort+Departure.Arrival.time.convenient+Food.and.drink+Gate.location+Inflight.wifi.service+Inflight.entertainment+Online.support+Ease.of.Online.booking+ On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Cleanliness+Online.boarding+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes,
                 ntree=200)#will take a while

predict2=predict(rf2, test1, predict.all=TRUE)
summary(predict2)

predict2$aggregate
cross2=table(test_target, predict2$aggregate)
cross2
(cross2[2,2]+cross2[1,1])/dim(test)[1]# hit rate #0.95193

#test_target    dissatisfied satisfied
#dissatisfied        16765       742
#satisfied            1131     20326

#############naiveBayes 
library(e1071)
naive_train=naiveBayes(factor(satisfaction)~.,data=train)
naive_train

######jaccard similarity index
naive_predict=predict(naive_train,test1, type="class")
dim(naive_predict)
crosstv=table(test_target, naive_predict)
crosstv[2,2]/(dim(test1)[1]-crosstv[1,1])#similarity#0.7075209
sum(diag(crosstv))/dim(test1)[1]#hit rate
(crosstv[2,2]+crosstv[1,1])/dim(test1)[1]#hit rate #0.8086695
crosstv
#naive_predict
#test_target    dissatisfied satisfied
#dissatisfied        13475      4032
#satisfied            3423     18034

############################crosstable
library(gmodels)
cross=data
head(cross)
cross$satisfaction=ifelse(cross$satisfaction=="satisfied",1,0)
cross$Customer.Type=ifelse(cross$Customer.Type=="Loyal Customer",1,0)
crosstable= CrossTable(cross$satisfaction,cross$Customer.Type)
summary(crosstable)
source("http://pcwww.liv.ac.uk/~william/R/crosstab.r")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Seat.comfort"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Departure.Arrival.time.convenient"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Food.and.drink"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Gate.location"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Inflight.wifi.service"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Inflight.entertainment"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Online.support"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Ease.of.Online.booking"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("On.board.service"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Leg.room.service"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Baggage.handling"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Checkin.service"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Cleanliness"), type = "c")
crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Online.boarding"), type = "c")
#crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Departure.Delay.in.Minutes"), type = "c")
#crosstab(cross, row.vars = c("satisfaction", "Customer.Type"), col.vars = c("Arrival.Delay.in.Minutes"), type = "c")
