
date()
#download.file(fileUrl, destfile = "./data/w1.1.csv") #erreur avec: , method = "curl"

setwd("/home/rstudio/projects/Personal-Activity-Prediction")

date()
train <- read.csv("pml-training.csv", header = T
                  , sep=","
                  ,na.strings = c(NA,"","NA","<NA>")
                  #, comment.char = ""
)
test <- read.csv("pml-testing.csv", header = T
                 , sep=","
                 ,na.strings = c(NA,"","NA","<NA>")
                 #, comment.char = ""
)
print(object.size(train),unit='Mb')

head(train)
names(train)
summary(train)

str(train)

# discard NAs
NAs <- apply(train,2,function(x) {sum(is.na(x))}) 
validData <- train[,which(NAs <= (10/100)*nrow(train))]
validData$X = NULL

# make training set
library(caret)
trainIndex <- createDataPartition(y = validData$classe, p=0.2,list=FALSE)
training <- validData[trainIndex,]
# make model

system.time(
        modFit <- train(classe ~.,data = training,method="gbm")
)
modFit

testing <- validData[-trainIndex,]
pred <- predict(modFit,testing)
table(pred,testing$classe)

sapply(ls(), function(x) {
        print(object.size(x),units='Mb')})
#rm()
gc()