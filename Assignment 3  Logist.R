
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","glmnet","MASS","e1071") #Check, and if needed install the necessary packages

bank<-read.csv(file.choose(), na.strings=c(""," ","NA"), header=TRUE) # Load the datafile to R

str(bank) # See if some data types were misclassified when importing data from CSV

# Fixing incorrectly classified data types:
bank$SEX <- as.factor(bank$SEX)
bank$EDUCATION<- as.factor(bank$EDUCATION)
bank$MARRIAGE <- as.factor(bank$MARRIAGE)
bank$PAY_1 <- as.factor(bank$PAY_1)
bank$PAY_2 <- as.factor(bank$PAY_2)
bank$PAY_3 <- as.factor(bank$PAY_3)
bank$PAY_4 <- as.factor(bank$PAY_4)
bank$PAY_5 <- as.factor(bank$PAY_5)
bank$PAY_6 <- as.factor(bank$PAY_6)
bank$default_0<-as.factor(bank$default_0)
bank$AGE<-scale(bank$AGE)
bank$BILL_AMT1<-scale(bank$BILL_AMT1)
bank$BILL_AMT2<-scale(bank$BILL_AMT2)
bank$BILL_AMT3<-scale(bank$BILL_AMT3)
bank$BILL_AMT4<-scale(bank$BILL_AMT4)
bank$BILL_AMT5<-scale(bank$BILL_AMT5)
bank$BILL_AMT6<-scale(bank$BILL_AMT6)

bank$PAY_AMT1<-scale(bank$PAY_AMT1)
bank$PAY_AMT2<-scale(bank$PAY_AMT2)
bank$PAY_AMT3<-scale(bank$PAY_AMT3)
bank$PAY_AMT4<-scale(bank$PAY_AMT4)
bank$PAY_AMT5<-scale(bank$PAY_AMT5)
bank$PAY_AMT6<-scale(bank$PAY_AMT6)

bank$LIMIT_BAL<-scale(bank$LIMIT_BAL)


#Feature Engineering

bank$Amount_Owed<-bank$BILL_AMT1+bank$BILL_AMT2+
  bank$BILL_AMT3+bank$BILL_AMT4+bank$BILL_AMT5+
  bank$BILL_AMT6-bank$PAY_AMT1-bank$PAY_AMT2-bank$PAY_AMT3-
  bank$PAY_AMT4-bank$PAY_AMT5-bank$PAY_AMT6

bank$AVG_Amount_Owed<-bank$Amount_Owed/6

bank$Payments_Missed<- ifelse(as.numeric(as.character(bank$PAY_1)) >=1,1,0)
bank$Payments_Missed<- ifelse(as.numeric(as.character(bank$PAY_2)) >=1,bank$Payments_Missed+1,bank$Payments_Missed)
bank$Payments_Missed<- ifelse(as.numeric(as.character(bank$PAY_3)) >=1,bank$Payments_Missed+1,bank$Payments_Missed)
bank$Payments_Missed<- ifelse(as.numeric(as.character(bank$PAY_4)) >=1,bank$Payments_Missed+1,bank$Payments_Missed)
bank$Payments_Missed<- ifelse(as.numeric(as.character(bank$PAY_5)) >=1,bank$Payments_Missed+1,bank$Payments_Missed)
bank$Payments_Missed<- ifelse(as.numeric(as.character(bank$PAY_6)) >=1,bank$Payments_Missed+1,bank$Payments_Missed)

bank$BalLim<- ((bank$BILL_AMT1+bank$BILL_AMT2+bank$BILL_AMT3+bank$BILL_AMT4+bank$BILL_AMT5+bank$BILL_AMT6)/6)/bank$LIMIT_BAL

#Apply combinerarecategories function to the data and then split it into testing and training data.

table(bank$Group.State)# check for rare categories

# Create another a custom function to combine rare categories into "Other."+the name of the original variavle (e.g., Other.State)
# This function has two arguments: the name of the dataframe and the count of observation in a category to define "rare"
combinerarecategories<-function(data_frame,mincount){ 
  for (i in 1 : ncol(data_frame)){
    a<-data_frame[,i]
    replace <- names(which(table(a) < mincount))
    levels(a)[levels(a) %in% replace] <-paste("Other",colnames(data_frame)[i],sep=".")
    data_frame[,i]<-a }
  return(data_frame) }

bank<-combinerarecategories(bank,20) #combine categories with <10 values in STCdata into "Other"


set.seed(1) #set a random number generation seed to ensure that the split is the same everytime


inTrain <- createDataPartition(y = bank$default_0,
                               p = 19200/24000, list = FALSE) #80/20
training <- bank[ inTrain,]
testing <- bank[ -inTrain,]

# Select the variables to be included in the "base-case" model
# First include all variables use glm(Retained.in.2012.~ ., data=training, family="binomial"(link="logit")) Then see which ones have "NA" in coefficients and remove those

model_logistic<-glm(default_0~., data=training, family="binomial"(link="logit"))

summary(model_logistic) 


## Stepwise regressions. There are three aproaches to runinng stepwise regressions: backward, forward and "both"
## In either approach we need to specify criterion for inclusion/exclusion. Most common ones: based on information criterion (e.g., AIC) or based on significance  
model_logistic_stepwiseAIC<-stepAIC(model_logistic,direction = c("both"),trace = 1) #AIC stepwise
summary(model_logistic_stepwiseAIC) 

par(mfrow=c(1,4))
plot(model_logistic_stepwiseAIC) #Error plots: similar nature to lm plots
par(mfrow=c(1,1))

###Finding predicitons: probabilities and classification
logistic_probabilities<-predict(model_logistic_stepwiseAIC,newdata=testing,type="response") #Predict probabilities
logistic_classification<-rep("1",4799) 
logistic_classification[logistic_probabilities<0.22108]="0" # Probability that a customer defaults
logistic_classification<-as.factor(logistic_classification)
summary(bank)

###Confusion matrix  
confusionMatrix(logistic_classification,testing$default_0,positive = "1") #Display confusion matrix

####ROC Curve
logistic_ROC_prediction <- prediction(logistic_probabilities, testing$default_0)
logistic_ROC <- performance(logistic_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(logistic_ROC_prediction,"auc") #Create AUC data
logistic_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
logistic_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(logistic_probabilities, testing$default_0, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

