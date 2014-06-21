require(caret) || install.packages("caret") 
require(gbm) || install.packages("gbm") 
require(e1071) || install.packages("e1071")

#library(caret)

date()
#download.file(fileUrl, destfile = "./data/w1.1.csv") #erreur avec: , method = "curl"

setwd("/home/rstudio/projects/Personal-Activity-Prediction")

rawtrain <- read.csv("pml-training.csv", header = T
                     , sep=","
                     ,na.strings = c(NA,"","NA","<NA>")
                     #, comment.char = ""
)
rawtest <- read.csv("pml-testing.csv", header = T
                    , sep=","
                    ,na.strings = c(NA,"","NA","<NA>")
                    #, comment.char = ""
)
print(object.size(rawtrain),unit='Mb')

head(rawtrain)
names(rawtrain)
summary(rawtrain)

str(rawtrain)

#PREPROCESSING

# discard unuseful variables

##NAs
NAs <- apply(rawtrain,2,function(x) {sum(is.na(x))}) #count NAs for each variable

##Other variables
#removeIndex <- grep("X|window|timestamp|user_name",names(rawtrain),value=T) #assume...
#removeIndex
#qplot(rawtrain$num_window,rawtrain$classe)
removeIndex <- grep("X|new_window|timestamp|user_name",names(rawtrain),value=F)
keepIndex <- grep("_x$|_y$|_z$",names(rawtrain),value=F)

##Remove all together here

###Option 1
fulltrain <- rawtrain[,-c(which(NAs > (10/100)*nrow(rawtrain)),removeIndex)] #-> 54 predictors
test <- rawtest[,-c(which(NAs > (10/100)*nrow(rawtrain)),removeIndex)]

###Option 2
#fulltrain <- rawtrain[,c(intersect(which(NAs <= (10/100)*nrow(rawtrain)),keepIndex),length(rawtrain))] #-> 37 predictors
#test <- rawtest[,c(intersect(which(NAs <= (10/100)*nrow(rawtrain)),keepIndex))]

##2nd (optional) pass: correlations NOT USED HERE
M <- abs(cor(fulltrain[,-which(names(fulltrain)=="classe")]))
diag(M) <- 0
w <- data.frame(which(M > 0.9,arr.ind=T))
w
apply(w,1,function(x) #todo, doesn't work
{names(fulltrain)[c(x$row,x$col)]}
)

##2nd (optional) pass: PCA both on train and validation data
fulltrain.preproc <- preProcess(fulltrain[,-which(names(fulltrain)=="classe")], method='pca', thresh=0.99)
fulltrain.pca <- predict(fulltrain.preproc, fulltrain[,-which(names(fulltrain)=="classe")])

test.preproc <- preProcess(test, method='pca', thresh=0.99)
test.pca <- predict(test.preproc, test)

##Split data into train + validation datas 
trainIndex <- createDataPartition(y = fulltrain$classe, p=0.7,list=FALSE)

part_training <- fulltrain[trainIndex,]
part_training.pca <- fulltrain.pca[trainIndex,]

part_validating <- fulltrain[-trainIndex,] #todo: change name to validation...
part_validating.pca <- fulltrain.pca[-trainIndex,] #todo: change name to validation...

# TRAINING
set.seed(123)

fitControl <- trainControl(method = "cv", number = 4) #speed up VS default parameter!

gbmGrid <-  expand.grid(interaction.depth = 1:3,
                        n.trees = (1:4)*50,
                        #shrinkage = c(0.1,0.5)
                        shrinkage = 0.1
)

system.time(
        #modFit <- train(classe ~. 
        modFit <- train(fulltrain[trainIndex,]$classe ~. #fulltrain instead of directly classe~. for compatibility with .pca datas
                        ,data = part_training #CHOOSE part_training.pca or part_training !
                        ,method="rf"
                        ,trControl = fitControl
                        ,verbose = FALSE
                        #,tuneGrid = gbmGrid #TO COMMENT IF METHOD IS NOT GBM !
                        #,allowParallel = T
        )
)
modFit
modFit$finalModel
#modFit$control$index
#modFit$resample
summary(modFit)
#summary(modFit)["user_name",]

length(unique(part_training$raw_timestamp_part_1))
sort(apply(part_training,2,function(x) {length(unique(x))}))
head(cbind(part_training$raw_timestamp_part_1,part_training$classe),50)

#VALIDATING MODEL
pred.part_validating <- predict(modFit,part_validating) #CHOOSE part_validating.pca or part_validating !
table(pred.part_validating,fulltrain[-trainIndex,]$classe)
confusionMatrix(pred.part_validating, part_validating$classe)

#TESTING MODEL
pred.test <- predict(modFit,test) #CHOOSE test.pca or test !
pred.test

#CLEAN MEMORY
sapply(ls(), function(x) {
        print(object.size(x),units='Mb')})
rm()
gc()

#PUT ANSWERS TO FILES
answers = as.character(pred.test)
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}
setwd("/home/rstudio/projects/Personal-Activity-Prediction/results")
pml_write_files(answers)