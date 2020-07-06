
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","glmnet","MASS","e1071")  #Check, and if needed install the necessary packages

# Load the data, correct mis-classified datafields, fixNAs -- same as you did in the logistic regression file
# To ensure "appled-to-apples" comparisons with logistic regression, use the same training and testing -- the code below only works in the same R session after you've ran the logistic regression code

###
### Support Vector Machines
###

model_svm <- svm(default_0 ~., data=training, probability=TRUE)
summary(model_svm)

svm_probabilities<-attr(predict(model_svm,newdata=testing, probability=TRUE), "prob")
svm_prediction<-svm_probabilities[,1]

svm_classification<-rep("1",4799)
svm_classification[svm_prediction<0.22108]="0" 
svm_classification<-as.factor(svm_classification)
confusionMatrix(svm_classification,testing$default_0,positive = "1")

####ROC Curve
svm_ROC_prediction <- prediction(svm_prediction, testing$default_0) #Calculate errors
svm_ROC_testing <- performance(svm_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(svm_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(svm_ROC_prediction,"auc") #Create AUC data
svm_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
svm_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(svm_prediction, testing$default_0, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_svm <- performance(svm_ROC_prediction,"lift","rpp")
plot(Lift_svm)
