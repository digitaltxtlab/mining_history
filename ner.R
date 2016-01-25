rm(list = ls())
library(rJava)
library(NLP)
library(openNLP)
library(RWeka)
library(magrittr)
filenames <- Sys.glob("C:/Users/KLN/Documents/textAnalysis/bible/newTestament/ntTotal/*.txt")

# import and preprocess data in pipe line
texts <- filenames %>%
  lapply(readLines) %>%
  lapply(paste0, collapse = " ") %>%
  lapply(as.String)
names(texts) <- basename(filenames)
# Extract entities from an AnnotatedPlainTextDocument
entities <- function(doc, kind) {
  s <- doc$content
  a <- annotations(doc)[[1]]
  if(hasArg(kind)) {
    k <- sapply(a$features, `[[`, "kind")
    s[a[k == kind]]
  } else {
    s[a[a$type == "entity"]]
  }
}

# create annotators
annotate_entities <- function(doc, annotation_pipeline) {
  annotations <- annotate(doc, annotation_pipeline)
  AnnotatedPlainTextDocument(doc, annotations)
}

text_pipeline <- list(
  Maxent_Sent_Token_Annotator(),
  Maxent_Word_Token_Annotator(),
  Maxent_Entity_Annotator(kind = "person"),
  Maxent_Entity_Annotator(kind = "location"),
  Maxent_Entity_Annotator(kind = "organization")
)

### annotate texts
text_annotated <- texts %>%
  lapply(annotate_entities, text_pipeline)

nchar(texts[[12]])
(text_annotated[12]$Acts.txt$content)
###### get named entities for historical documents
k <- c(12,18,20,21,22)# historical books
histfilenames <- filenames[k]
  histfilenames <-gsub(".txt", "", histfilenames)
histPerson <- histOrganization <- histLocation <- list()
histMat <- matrix(0,nrow = length(k), ncol = 4)
colnames(histMat) <- c('Person','Organization','Location','chars')
rownames(histMat) <- basename(histfilenames)
for (i in 1:length(k)){
  histPerson[[i]] <- entities(text_annotated[[k[i]]],kind = "person")
  histOrganization[[i]] <- entities(text_annotated[[k[i]]],kind = "organization") 
  histLocation[[i]] <- entities(text_annotated[[k[i]]],kind = "location")
  histMat[i,1] <- length(histPerson[[i]])
  histMat[i,2] <- length(histOrganization[[i]])
  histMat[i,3] <- length(histLocation[[i]])
  histMat[i,4] <- nchar(texts[[k[i]]])
}
# get relative distribution of entities
histRelativeMat <- matrix(0,nrow = length(k), ncol = 4)
rownames(histRelativeMat) <- rownames(histMat)
colnames(histRelativeMat) <- c('Person','Organization','Location','chars')
for (i in 1:4){
  histRelativeMat[,i] <- histMat[,i]/colSums(histMat)[i]
}
### export data
write.csv(histRelativeMat,file = "histRelativeMat.csv")
