rm(list = ls())
load("nt.RData")


library(plyr)
library(topicmodels)
k = 20
progress.bar <- create_progress_bar("text")
progress.bar$init(k)
best.mdl <- list()
for(i in 2:k){
  best.mdl[[i-1]] <- LDA(cpdtm, i)
  progress.bar$step()
}
best.mdl.logL <- as.data.frame(as.matrix(lapply(best.mdl, logLik)))
best.mdl.logL.df <- data.frame(topics=c(2:k), LL=as.numeric(as.matrix(best.mdl.logL)))
dev.new()
plot(best.mdl.logL.df$topics,best.mdl.logL.df$LL,main = 'Parameter estimation', xlab = 'Topic', ylab = 'Log liklihood of model')

resultMat <- matrix(c(best.mdl.logL.df$topics,best.mdl.logL.df$LL),ncol = 2)
# save(resultMat, file = "logliktopics.RData")
setwd(dd)
load("logliktopics.RData")
write.table(resultMat, file = "nttloglik.csv", sep = ",", col.names = FALSE,row.names = FALSE,
            qmethod = "double")
#####
library(tm)
# dtm
X <- cpdtm 
dim(X)

library("topicmodels")
k <- 20
SEED <- 1234
X_TM <- list(VEM = LDA(X, k = k, control = list(seed = SEED)),
             VEM_fixed = LDA(X, k = k, control = list(estimate.alpha = FALSE, seed = SEED)),
             Gibbs = LDA(X, k = k, method = "Gibbs", control = list(seed = SEED, burnin = 1000, thin = 100, iter = 1000)),
             CTM = CTM(X, k = k, control = list(seed = SEED, var = list(tol = 10^-4), em = list(tol = 10^-3))))

save(X_TM, file = "ntTopicModels.RData")

load(file = "ntTopicModels.RData")
# most likely topics for documents
Topic <- topics(X_TM[["VEM"]], 1)
# ranked terms for topics
Terms <- terms(X_TM[["VEM"]], 10)
Terms

x <-posterior(X_TM[["VEM"]])$topics["Matthew",]
plot(x)
y <- posterior(X_TM[["VEM"]])$topics["Luke",]
dev.new
plot(y)

write.csv(x,file = "matthewTopics.csv",row.names = TRUE)







