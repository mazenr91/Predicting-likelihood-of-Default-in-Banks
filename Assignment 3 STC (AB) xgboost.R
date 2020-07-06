
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","xgboost") #Check, and if needed install the necessary packages

# Load the data, correct mis-classified datafields, fixNAs -- same as you did in the logistic regression file
# To ensure "appled-to-apples" comparisons with logistic regression, use the same training and testing -- the code below only works in the same R session after you've ran the logistic regression code

###
### Gradient Boosting Machines (XGboost)
###

# As many other methods, xgboost requires matrices as inputs
# But creating them per the two lines below is a BAD idea: may lead to different levels

# x_train <-model.matrix(Retained.in.2012.~ ., data = training)
# x_test <-model.matrix(Retained.in.2012.~ ., data = testing)

bank_matrix <- model.matrix(default_0~ ., data = bank)[,-1]

x_train <- bank_matrix[ inTrain,]
x_test <- bank_matrix[ -inTrain,]

y_train <-training$default_0
y_test <-testing$default_0

model_XGboost<-xgboost(data = data.matrix(x_train), 
                       label = as.numeric(as.character(y_train)), 
                       eta = 0.1,       # hyperparameter: learning rate 
                       max_depth = 20,  # hyperparameter: size of a tree in each boosting iteration
                       nround=50,       # hyperparameter: number of boosting iterations  
                       objective = "binary:logistic"
)

XGboost_prediction<-predict(model_XGboost,newdata=x_test, type="response") #Predict classification (for confusion matrix)
confusionMatrix(as.factor(ifelse(XGboost_prediction>0.22108,1,0)),y_test,positive="1") #Display confusion matrix

####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction, y_test) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(XGboost_prediction, y_test, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_XGboost <- performance(XGboost_ROC_prediction,"lift","rpp")
plot(Lift_XGboost)
