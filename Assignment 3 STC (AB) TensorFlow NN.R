if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("tensorflow", "keras", "dplyr")

# Load the data, correct mis-classified datafields, fixNAs -- same as you did in the logistic regression file
# To ensure "appled-to-apples" comparisons with logistic regression, use the same training and testing -- the code below only works in the same R session after you've ran the logistic regression code

###
### Deep Learning with TensorFlow
###

# The initial installation is a bit tricky, use the code below for the first time; after that, pacman

install.packages("tensorflow")
library(tensorflow)
install_tensorflow()

install.packages("keras")
library(keras)
install_keras()

# Preprocessing data for inputting into Keras
# Tensors are matrices... hence the input data has to be in a form of a matrix

bank_matrix <- model.matrix(default_0~ ., data = bank)[,-1]

bank_matrix <- scale(bank_matrix) # scaling for X

x_train <- bank_matrix[ inTrain,]
x_test <- bank_matrix[ -inTrain,]

y_train <-training$default_0
y_test <-testing$default_0

x_train <- array_reshape(x_train, c(nrow(x_train), 67)) #Keras interprets data using row-major semantics (as opposed to R's default column-major semantics). Hence need to "reshape" the matrices 
x_test <- array_reshape(x_test, c(nrow(x_test), 67))

y_train <- to_categorical(y_train, 2) # converting to categorical for Y
y_test <- to_categorical(y_test, 2) # converting to categorical for Y
use_condaenv("r-tensorflow")
#
# Defining the neural network model architecture: layers, units, activations. 
#

# common kinds of layers: https://keras.io/layers/about-keras-layers/

# dense -- connect to each neuron
# dropout -- connect to each neuron with some probability 
# convolution -- foundation of computer vision/perception 
# recurrent -- foundation of time-dependent modeling (text, time-series, etc.) Wekaer than LSTM
# LSTM -- long short-term memory (time-series, semantics)
# flatten/embedding/ -- utility layers: particular kinds of data preparation 

# common kinds of activations: https://keras.io/activations/
# relu -- piece-wise linear 
# sigmoid, tanh -- S-shaped 
# softmax -- normalization to probabilities using exp/sum(exp) transform [like in logistic regression]

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2, activation = 'softmax')

#
# Compiling the model 
#

# common loss functions: https://keras.io/losses/
# mean_absolute_percentage_error, mean_absolute_error -- for continuous quantities
# binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy -- for events (binary, multinomial)

# common optimizers: https://keras.io/optimizers/
# adam -- commonly used
# SGD -- "stochastic gradient descent"

# common metrics: https://keras.io/metrics/ 
# accuracy, mae, mape 

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# Training / "fitting" the model

history <- model %>% fit(
  x_train, y_train, # on what data to train
  epochs = 30, # how many repetitions to have
  batch_size = 128, # how many datapoints are fed to the network at a time 
  validation_split = 0.2  # percentage of training data to keep for cross-validation 
)

summary(model)

plot(history)

# model %>% evaluate(x_test, y_test) # apply the model to testing data

TF_NN_probabilities <- model %>% predict(x_test)  # predict probabilities
TF_NN_prediction <- TF_NN_probabilities[,2]

TF_NN_classification<-rep("1",500)
TF_NN_classification[TF_NN_prediction<0.6073]="0" 
TF_NN_classification<-as.factor(TF_NN_classification)

confusionMatrix(TF_NN_classification,testing$Retained.in.2012.,positive = "1")

####ROC Curve
TF_NN_ROC_prediction <- prediction(TF_NN_prediction, testing$Retained.in.2012.) #Calculate errors
TF_NN_ROC_testing <- performance(TF_NN_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(TF_NN_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(TF_NN_ROC_prediction,"auc") #Create AUC data
TF_NN_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
TF_NN_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(TF_NN_prediction, testing$Retained.in.2012., cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_TF_NN <- performance(TF_NN_ROC_prediction,"lift","rpp")
plot(Lift_TF_NN)
