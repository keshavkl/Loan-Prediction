library(readr)
library(ggplot2)
library(mice)
library(xgboost)
library(Matrix)


train <- read_csv("train.csv")
test <- read_csv("test.csv")

status.y <- train$Loan_Status
train$Loan_Status <- NULL
status.y[status.y == "Y"] <- 1
status.y[status.y == "N"] <- 0


all <- rbind(train, test)
all$Loan_ID <- NULL

char <- lapply(all, class) == "character"
all[, char] <- lapply(all[, char], as.factor)

all$Dependents <- as.factor(all$Dependents)
all$Loan_Amount_Term <- as.factor(all$Loan_Amount_Term)
all$Credit_History <- as.factor(all$Credit_History)

impdata <- mice(all, m = 5, maxit = 10)
mydata <- complete(impdata, 5)

train_new <- mydata[1:nrow(train), ]
test_new <- mydata[-(1:nrow(train)), ]

train_new$Loan_Status <- status.y

train_new <- sparse.model.matrix(Loan_Status ~ ., data = train_new)

dtrain <- xgb.DMatrix(data=train_new, label = status.y)

watchlist <- list(train_new = dtrain)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.03,
                max_depth           = 4,
                subsample           = 0.6,
                colsample_bytree    = 0.75,
                early.stop.round    = 1
)

xcv <- xgb.cv(  params = param,
                data = dtrain,
                nrounds = 500,
                nfold = 2,
                metrics = {'auc'}
)
which.max(xcv$test.auc.mean)
plot(xcv$test.auc.mean)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 135, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)
importance_matrix <- xgb.importance(train_new@Dimnames[[2]], model = clf)
xgb.plot.importance(importance_matrix)

tpred <- predict(clf, train_new)
table(round(tpred), status.y)
sum(diag(table(round(tpred), status.y)))/sum(table(round(tpred), status.y))

test_new$Loan_Status <- -1
test_new <- sparse.model.matrix(Loan_Status ~ ., data = test_new)

preds <- predict(clf, test_new)
mypreds <- preds
mypreds[mypreds >= 0.40] <- "Y"
mypreds[mypreds < 0.40] <- "N"
submission <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = mypreds)
write.csv(submission, "submission_40thr.csv", row.names = F)











