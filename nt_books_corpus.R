rm(list = ls())
dd = "C:/Users/KLN/Documents/textAnalysis/bible/newTestament/ntTotal";
library(tm)
c  <- Corpus(DirSource(dd, encoding = "UTF-8"), readerControl = list(language="PlainTextDocument"))
names(c) <- gsub("\\..*","",names(c))
filenames <- names(c)

cdtm <- DocumentTermMatrix(c)
dim(cdtm)
#dim(cdtm)[1]*dim(cdtm)[2]
#cdtmmat <- as.matrix(cdtm)
#sum(colSums(cdtmmat))
#cdtmmat[,'wine']

## simple transformations
c <- tm_map(c, removePunctuation)
c <- tm_map(c, removeNumbers) 
c <- tm_map(c, content_transformer(tolower))
cdtm <- DocumentTermMatrix(c)
dim(cdtm)

# with stopwords
c <- tm_map(c, removeWords, stopwords("english"))
cdtm <- DocumentTermMatrix(c)
dim(cdtm)
stopwords('English')


## stemming
cmat <- as.matrix(cdtm)
# find word forms
idx <- which(colnames(cmat) == 'savior')
idxs <- (idx-1):(idx+10)
colnames(cmat)[idxs]

c <- tm_map(c, stripWhitespace)
c <- tm_map(c, stemDocument, language = "english")# Porter's algorithm
cdtm <- DocumentTermMatrix(c)
dim(cdtm)
cmat <- as.matrix(cdtm)
# find word forms
idx <- which(colnames(cmat) == 'pray')
idxs <- (idx-1):(idx+10)
colnames(cmat)[idxs]

# synonym identification
install.packages("wordnet", dependencies = TRUE)

library("wordnet")# load wordnet
setDict("C:/Program Files (x86)/WordNet/2.1/dict")# set dictionary

synonyms('lord','NOUN')
synonyms('christ','NOUN')


### preprocessing
cp <- tm_map(c, removePunctuation)
cp <- tm_map(cp, removeNumbers) 
cp <- tm_map(cp, content_transformer(tolower))
cp <- tm_map(cp, removeWords, stopwords("english"))
cp <- tm_map(cp, stripWhitespace)
cp <- tm_map(cp, stemDocument, language = "english")# Porter's algorithm
cp <- tm_map(cp, PlainTextDocument)
names(cp) <- names(c)
### build document term matrix
cpdtm <- DocumentTermMatrix(cp)
dim(cpdtm)
rownames(cpdtm) <- names(c)
dim(cpdtm)
## reduce sparsity
docsparse <- function(mindocs,dtm){
  n = length(row.names(dtm))
  sparse <- 1 - mindocs/n;
  dtmreduce <- removeSparseTerms(dtm, sparse)
  return(dtmreduce)
}
cpdtm <- docsparse(3,cpdtm)
dim(cpdtm)




1-(3/27)


# transform to matrix
library(reshape2)
cpmat <- as.matrix(cpdtm)

write.table(cpmat, file = "preprocessedDtm.csv", sep = ",", row.names = TRUE,
            col.names = TRUE)
write.table(colnames(cpmat), file = "voc1537.csv", sep = ",")

cpmat[,'jesus']
find(colnames(cpmat) == 'jesus')
cpmat[,'christ']

which(colnames(cpmat) == 'jesus')
which(colnames(cpmat) == 'said')

which(colnames(cpmat) == 'jesus')
which(colnames(cpmat) == 'christ')


cpmatdense <- melt(cpmat, value.name = "count")
# plot number of words and document length
dev.new()
doclen <- sort(rowSums(cpmat), decreasing = TRUE)
wordfreq <- sort(colSums(cpmat), decreasing = TRUE)
head(wordfreq)
par(mfrow=c(1,2))
plot(log(1:length(wordfreq)),log(wordfreq), main = "Vocabulary", xlab = "Index", ylab = "Word frequency",bty="n")
plot(doclen, main = "Document length", ylab = "Number of words", bty="n")
text(1:length(doclen),doclen,names(doclen))

# word frequencies and document length
wordfreq[1:50]
doclen

# save.image(file = "nt.RData")

############# partitioning
### partional prototype-based exclusive clustering
norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)#
# %%% explain function object
cpmatnorm <- norm_eucl(cpmat)

# distance matrix
?dist

dcpmatnorm <- dist(cpmatnorm, method = "euclidean",diag = TRUE, upper = TRUE)
dcpmatnorm
# graphical approach to estimate optimal number of groups
wssplot <- function(data, nc=15, seed=1234){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}
maxk = 20
dev.new()
wssplot(cpmatnorm,nc = maxk) 
# k-means on length normalized data with k = 4
k = 26
cpcl <- kmeans(cpmatnorm, k)
# inspect cluster object
cpcl
# classification
cpcl$cluster
# goodness of the classification
# %%% explain BSS/TSS
as.numeric(cpcl[6])/as.numeric(cpcl[3])# c
# plot clusters using the first 2 principal components
dev.new()
plot(prcomp(cpmatnorm)$x, col=cpcl$cl)

text(prcomp(cpmatnorm)$x[,1],prcomp(cpmatnorm)$x[,2],rownames(cpmat))

clexport = matrix(c(prcomp(cpmatnorm)$x[,1],prcomp(cpmatnorm)$x[,2],cpcl$cl),ncol = 3)
rownames(clexport) <- rownames(cpmat)
write.csv(clexport,file = "clexport.csv",row.names = TRUE)

### embedded
library(proxy)
write.csv(cpmat,file = "cpmat.csv",row.names = TRUE,col.names = TRUE)
colnames(cpmat)
rownames(cpmat)


cpdist <- dist(cpmat, method="cosine")# use dot product and euclid dist as length normalizer
cphc <- hclust(cpdist, method="average")
# plot with dendrogram
dev.new()
plot(cphc)

cphc$height
cphc$order


