# This is the set of commands used for the Statistical
# Learning with caret class given on May 9, 2019.

# Source for code used below is from 
# https://topepo.github.io/caret/index.html

# We will be learning about the caret package.

# Install the packages with dependencies.
install.packages("caret", dependencies = c("Depends"))
install.packages("AppliedPredictiveModeling", dependencies = c("Depends"))
install.packages("mlbench", dependencies = c("Depends"))

# Load the packages
library(caret)
library(AppliedPredictiveModeling)
library(mlbench)


###############################
#    Visualizations
###############################

# We will use Fisher's iris data to illustrate classification and
# the Boston housing data to explore regression.

# Let's look at these two data sets.

# str() gives information about the structure of the data object.

str(iris)

data("BostonHousing")
str(BostonHousing)

# Let's use feature plot, which is part of the caret package.
# It is a wrapper for lattice plots of predictgor variables.

transparentTheme(trans = .4)
featurePlot(x = iris[, 1:4], 
            y = iris$Species, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

# Add ellipses.

featurePlot(x = iris[, 1:4], 
            y = iris$Species, 
            plot = "ellipse",
            ## Add a key at the top
            auto.key = list(columns = 3))

# Density plots

transparentTheme(trans = .9)
featurePlot(x = iris[, 1:4], 
            y = iris$Species,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(4, 1), 
            auto.key = list(columns = 3))


# Box plots

featurePlot(x = iris[, 1:4], 
            y = iris$Species, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(4,1 ), 
            auto.key = list(columns = 2))

# Now for the Boston Housing Data - regression application

library(mlbench)
data(BostonHousing)

# Set the predictors - just for the plot examples
regVar <- c("age", "lstat", "tax")

# Create scatterplots of each predictor with response variable medv. 

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
featurePlot(x = BostonHousing[, regVar], 
            y = BostonHousing$medv, 
            plot = "scatter", 
            layout = c(3, 1))

# Add smoother to better see the relationships.

featurePlot(x = BostonHousing[, regVar], 
            y = BostonHousing$medv, 
            plot = "scatter",
            type = c("p", "smooth"),
            span = .5,
            layout = c(3, 1))

##############################################
#   Create Training and Testing Data Sets
##############################################

# First we do this for the iris data.

# Get index of observations in training set.
trainIndex <- createDataPartition(iris$Species, p = .8, 
                                  list = FALSE, 
                                  times = 1)


# Create the training and testing data objects.
# This is all the observations in training set.
irisTrain <- iris[ trainIndex,]

# Get all of the rest - not in training set.
irisTest  <- iris[-trainIndex,]

# Let's explore the two sets:
str(irisTrain)
summary(irisTrain)

str(irisTest)
summary(irisTest)

#  Now for the Boston Housing data

trainIndex <- createDataPartition(BostonHousing$medv, p = .75, 
                                  list = FALSE, 
                                  times = 1)


# Create the training and testing data objects.

bhTrain <- BostonHousing[ trainIndex,]
bhTest  <- BostonHousing[-trainIndex,]

# Explore the two data sets

str(bhTrain)
summary(bhTrain)
str(bhTest)
summary(bhTest)



###############################################
#   Classification
###############################################

# Let's use linear and quadratic classifiers for the iris data.
# These would be the lda and the qda models.

# See http://uc-r.github.io/discriminant_analysis 

# Use trainControl() to adjust the resampling method.

ctrl <- trainControl(method = "repeatedcv", repeats = 3)

ldaFit <- train(
  Species ~.,
  data = irisTrain,
  method = "lda"
)

# Assess results using the Test data set
table(irisTest$Species, predict(ldaFit,irisTest[,1:4]))

# YOU TRY THIS, see the results using the training set.
table(irisTrain$Species, predict(ldaFit))

# Now, do the same things with quadratic classifier.

qdaFit <- train(
  Species ~.,
  data = irisTrain,
  method = "qda"
)

# Assess results using the Test data set
table(irisTest$Species, predict(qdaFit,irisTest[,1:4]))

# YOU TRY THIS, see the results using the training set.
table(irisTrain$Species, predict(qdaFit))

######################################
#   Regression example
######################################

# We will use all of the Boston Housing data first.

lmFit <- train(medv ~ .,
               data = BostonHousing,
               method = "lm")

summary(lmFit)

# Can look at varable importance
varImp(lmFit)

# You can plot the variable importance using.
plot(varImp(lmFit))

# Run some diagnostic plots.

# Usual residual plot
residuals<-resid(lmFit)
predictedValues<-predict(lmFit)
plot(BostonHousing$medv,residuals)
abline(0,0)

# Plot of actual vs predicted
plot(BostonHousing$medv,predictedValues)

# Look at mean squared error and R-squared using test set.

# Get the predicted values using the model on the test set.
predictedVal<-predict(lmFit,bhTest)

modelvalues<-data.frame(obs = bhTest$medv, pred=predictedVal)

# Let's look at the observed and predicted values.
head(modelvalues)

# Gives the MSE and R-2 and MAE
defaultSummary(modelvalues)

# YOU TRY IT: Use different linear models. How are the results?

# YOU TRY IT: Repeat the above analyses using the bhTrain data. 
# Are your results/models/decisions different?
