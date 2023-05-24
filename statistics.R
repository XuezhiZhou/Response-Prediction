setwd("E:/project/pNR_rectalcancer/attention_radiomics/")
library(openxlsx)
library(rms)
library(Hmisc)
library(glmnet)
library(glm2)
library(pROC)
library(caret)
library(e1071)
library(reshape2)
library(ggplot2)
library(bootstrap) 
library(nnet)
library(stringr)
library(PredictABEL)

train_clinical_all<-read.xlsx("E:/project/pNR_rectalcancer/attention_radiomics/whole.xlsx",sheet = 'train',startRow = 1,colNames = TRUE,rowNames = TRUE)
validation_clinical_all<-read.xlsx("E:/project/pNR_rectalcancer/attention_radiomics/whole.xlsx",sheet = 'validation',startRow = 1,colNames = TRUE,rowNames = TRUE)
test_clinical_all<- read.xlsx("E:/project/pNR_rectalcancer/attention_radiomics/whole.xlsx",sheet = 'test',startRow = 1,colNames = TRUE,rowNames = TRUE)

#############ROC CFDL+CFRS
train_clinical_all$CFDL_CFRS<-train_clinical_all$CFDL+train_clinical_all$CFRS
validation_clinical_all$CFDL_CFRS<-validation_clinical_all$CFDL+validation_clinical_all$CFRS
test_clinical_all$CFDL_CFRS<-test_clinical_all$CFDL+test_clinical_all$CFRS

CFDL_CFRS_train <- roc(label ~ CFDL_CFRS,data=train_clinical_all, ci=TRUE)
CFDL_CFRS_validation <- roc(label ~ CFDL_CFRS,data=validation_clinical_all, ci=TRUE)
CFDL_CFRS_test <- roc(label ~ CFDL_CFRS,data=test_clinical_all, ci=TRUE)

plot(CFDL_CFRS_train,col='red',lwd=3,lty = 1, main="CFDL + CFRS",print.thres=F,legacy.axes=TRUE) #
plot.roc(CFDL_CFRS_validation,add=TRUE,col='blue',lty = 1,lwd=3) #F0E442
plot.roc(CFDL_CFRS_test,add=TRUE,col='green',lty = 1,lwd=3) #F0E442
legend("bottomright",cex=1.1,bty="n",legend=c(paste("Training Cohort: ",signif(CFDL_CFRS_train$auc,3)),
                                              paste("Internal Validation Cohort: ",signif(CFDL_CFRS_validation$auc,3)),
                                              paste("External Validation Cohort: ",signif(CFDL_CFRS_test$auc,3))),
       col=c("red","blue","green"),lty = c (1,1,1),lwd=3)
#############CFDL+CFRS
Youden<-CFDL_CFRS_train$specificities+CFDL_CFRS_train$sensitivities
thres_CFDL_CFRS <- sort(train_clinical_all$CFDL_CFRS)[which(Youden==max(Youden))[1]] #youden cutoff
TABLE_train<-table(1*((train_clinical_all$CFDL_CFRS)>=thres_CFDL_CFRS),train_clinical_all$label)
inf<-confusionMatrix(TABLE_train, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(CFDL_CFRS_train$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')
 
TABLE_validation<-table(1*((validation_clinical_all$CFDL_CFRS)>=thres_CFDL_CFRS),validation_clinical_all$label)
inf<-confusionMatrix(TABLE_validation, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(CFDL_CFRS_validation$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')
 
TABLE_test<-table(1*((test_clinical_all$CFDL_CFRS)>=thres_CFDL_CFRS),test_clinical_all$label)
inf<-confusionMatrix(TABLE_test, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(CFDL_CFRS_test$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')

#############ROC CFDL
CFDL_train <- roc(label ~ CFDL,data=train_clinical_all, ci=TRUE)
CFDL_validation <- roc(label ~ CFDL,data=validation_clinical_all, ci=TRUE)
CFDL_test <- roc(label ~ CFDL,data=test_clinical_all, ci=TRUE)

plot(CFDL_train,col='red',lwd=3,lty = 1, main="CFDL",print.thres=F,legacy.axes=TRUE) #
plot.roc(CFDL_validation,add=TRUE,col='blue',lty = 1,lwd=3) #F0E442
plot.roc(CFDL_test,add=TRUE,col='green',lty = 1,lwd=3) #F0E442
legend("bottomright",cex=1.1,bty="n",legend=c(paste("train: ",signif(CFDL_train$auc,3)),
                                              paste("validation: ",signif(CFDL_validation$auc,3)),
                                              paste("test: ",signif(CFDL_test$auc,3))),
       col=c("red","blue","green"),lty = c (1,1,1),lwd=3)
#############CFDL
Youden<-CFDL_train$specificities+CFDL_train$sensitivities
thres_CFDL <- sort(train_clinical_all$CFDL)[which(Youden==max(Youden))[1]]
TABLE_train<-table(1*((train_clinical_all$CFDL)>=thres_CFDL),train_clinical_all$label)
inf<-confusionMatrix(TABLE_train, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(CFDL_train$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')
 
TABLE_validation<-table(1*((validation_clinical_all$CFDL)>=thres_CFDL),validation_clinical_all$label)
inf<-confusionMatrix(TABLE_validation, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(CFDL_validation$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')

TABLE_test<-table(1*((test_clinical_all$CFDL)>=thres_CFDL),test_clinical_all$label)
inf<-confusionMatrix(TABLE_test, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(CFDL_test$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')
############################################
#############ROC DL
DL_train <- roc(label ~ DL,data=train_clinical_all, ci=TRUE)
DL_validation <- roc(label ~ DL,data=validation_clinical_all, ci=TRUE)
DL_test <- roc(label ~ DL,data=test_clinical_all, ci=TRUE)

plot(DL_train,col='red',lwd=3,lty = 1, main="DL",print.thres=F,legacy.axes=TRUE) #
plot.roc(DL_validation,add=TRUE,col='blue',lty = 1,lwd=3) #F0E442
plot.roc(DL_test,add=TRUE,col='green',lty = 1,lwd=3) #F0E442
legend("bottomright",cex=1.1,bty="n",legend=c(paste("train: ",signif(DL_train$auc,3)),
                                              paste("validation: ",signif(DL_validation$auc,3)),
                                              paste("test: ",signif(DL_test$auc,3))),
       col=c("red","blue","green"),lty = c (1,1,1),lwd=3)
############# DL
Youden<-DL_train$specificities+DL_train$sensitivities
thres_DL <- sort(train_clinical_all$DL)[which(Youden==max(Youden))[1]]
TABLE_train<-table(1*((train_clinical_all$DL)>=thres_DL),train_clinical_all$label)
inf<-confusionMatrix(TABLE_train, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(DL_train$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')

TABLE_validation<-table(1*((validation_clinical_all$DL)>=thres_DL),validation_clinical_all$label)
inf<-confusionMatrix(TABLE_validation, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(DL_validation$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')

TABLE_test<-table(1*((test_clinical_all$DL)>=thres_DL),test_clinical_all$label)
inf<-confusionMatrix(TABLE_test, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(DL_test$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')
############################################
#############ROC CFRS
CFRS_train <- roc(label ~ CFRS,data=train_clinical_all, ci=TRUE)
CFRS_validation <- roc(label ~ CFRS,data=validation_clinical_all, ci=TRUE)
CFRS_test <- roc(label ~ CFRS,data=test_clinical_all, ci=TRUE)

plot(CFRS_train,col='red',lwd=3,lty = 1, main="CFRS",print.thres=F,legacy.axes=TRUE) #
plot.roc(CFRS_validation,add=TRUE,col='blue',lty = 1,lwd=3) #F0E442
plot.roc(CFRS_test,add=TRUE,col='green',lty = 1,lwd=3) #F0E442
legend("bottomright",cex=1.1,bty="n",legend=c(paste("train: ",signif(CFRS_train$auc,3)),
                                              paste("validation: ",signif(CFRS_validation$auc,3)),
                                              paste("test: ",signif(CFRS_test$auc,3))),
       col=c("red","blue","green"),lty = c (1,1,1),lwd=3)
#############CFRS
Youden<-CFRS_train$specificities+CFRS_train$sensitivities
thres_CFRS <- sort(train_clinical_all$CFRS)[which(Youden==max(Youden))[1]]	#youden cutoff
TABLE_train<-table(1*((train_clinical_all$CFRS)>=thres_CFRS),train_clinical_all$label)
inf<-confusionMatrix(TABLE_train, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(CFRS_train$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')

TABLE_validation<-table(1*((validation_clinical_all$CFRS)>=thres_CFRS),validation_clinical_all$label)
inf<-confusionMatrix(TABLE_validation, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(CFRS_validation$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')

TABLE_test<-table(1*((test_clinical_all$CFRS)>=thres_CFRS),test_clinical_all$label)
inf<-confusionMatrix(TABLE_test, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(CFRS_test$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')
############################################
#############ROC RS
RS_train <- roc(label ~ RS,data=train_clinical_all, ci=TRUE)
RS_validation <- roc(label ~ RS,data=validation_clinical_all, ci=TRUE)
RS_test <- roc(label ~ RS,data=test_clinical_all, ci=TRUE)

plot(RS_train,col='red',lwd=3,lty = 1, main="RS",print.thres=F,legacy.axes=TRUE) #
plot.roc(RS_validation,add=TRUE,col='blue',lty = 1,lwd=3) #F0E442
plot.roc(RS_test,add=TRUE,col='green',lty = 1,lwd=3) #F0E442
legend("bottomright",cex=1.1,bty="n",legend=c(paste("train: ",signif(RS_train$auc,3)),
                                              paste("validation: ",signif(RS_validation$auc,3)),
                                              paste("test: ",signif(RS_test$auc,3))),
       col=c("red","blue","green"),lty = c (1,1,1),lwd=3)
############# RS
Youden<-RS_train$specificities+RS_train$sensitivities
thres_RS <- sort(train_clinical_all$RS)[which(Youden==max(Youden))[1]]
TABLE_train<-table(1*((train_clinical_all$RS)>=thres_RS),train_clinical_all$label)
inf<-confusionMatrix(TABLE_train, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(RS_train$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')

TABLE_validation<-table(1*((validation_clinical_all$RS)>=thres_RS),validation_clinical_all$label)
inf<-confusionMatrix(TABLE_validation, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(RS_validation$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')
 
TABLE_test<-table(1*((test_clinical_all$RS)>=thres_RS),test_clinical_all$label)
inf<-confusionMatrix(TABLE_test, positive = "1")
list<-vector(mode="numeric",length=0)
cat('\n',"auc:",signif(RS_test$auc,3),'\n',
    "acc:",inf[[3]][1],'\n',
    "sen:",inf[[4]][1],'\n',
    "spe:",inf[[4]][2],'\n',
    "ppv:",inf[[4]][3],'\n',
    "npv:",inf[[4]][4],'\n')

###############################plot
library(cowplot)
library(dplyr)
library(readr)
source("R_rainclouds.R")
raincloud_data<-read.xlsx("E:/project/pNR_rectalcancer/attention_radiomics/whole.xlsx",sheet = 'train_raincloud',startRow = 1,colNames = TRUE,rowNames = FALSE)
raincloud_data$label<-factor(raincloud_data$label)
ggplot(raincloud_data, aes(x = model, y = probability, fill = label))+
  geom_flat_violin(aes(fill = label),position = position_nudge(x = .18, y = 0), adjust = 1.0, trim = TRUE, alpha = 1, colour = NA)+
  geom_point(aes(x = model, y = probability, colour = label),position = position_jitterdodge(jitter.width = 0.2, jitter.height = 0, dodge.width = 0.25), size = 2, shape = 20)+
  geom_boxplot(aes(x = model, y = probability, fill = label),position = position_dodge(width = 0.25), outlier.shape = NA, alpha = 1, width = .2, colour = "black")+
  # scale_colour_brewer(palette = "Set1")+
  # scale_fill_brewer(palette = "Set1")+
  scale_fill_manual(values = c("red", "blue"))+
  scale_color_manual(values = c("red", "blue"))+
  theme_bw()+theme(panel.border = element_blank(),
                   panel.grid.major = element_blank(),
                   panel.grid.minor = element_blank(),
                   axis.line = element_line(colour = "black",size=1),
                   axis.ticks = element_line(colour = "black",size=1),
                   axis.text = element_text(size = 15,colour = "black"),
                   axis.title = element_text(size = 20),
                   plot.title = element_text(size = 20,hjust = 0.5),
                   legend.position="top")+
  labs(title = "Training cohort", x = 'Model', y = "Probability")

raincloud_data<-read.xlsx("E:/project/pNR_rectalcancer/attention_radiomics/whole.xlsx",sheet = 'validation_raincloud',startRow = 1,colNames = TRUE,rowNames = FALSE)
raincloud_data$label<-factor(raincloud_data$label)
ggplot(raincloud_data, aes(x = model, y = probability, fill = label)) +
  geom_flat_violin(aes(fill = label),position = position_nudge(x = .18, y = 0), adjust = 1.0, trim = TRUE, alpha = 1, colour = NA)+
  geom_point(aes(x = model, y = probability, colour = label),position = position_jitterdodge(jitter.width = 0.2, jitter.height = 0, dodge.width = 0.25), size = 2, shape = 20)+
  geom_boxplot(aes(x = model, y = probability, fill = label),position = position_dodge(width = 0.25), outlier.shape = NA, alpha = 1, width = .2, colour = "black")+
  # scale_colour_brewer(palette = "Set1")+
  # scale_fill_brewer(palette = "Set1")+
  scale_fill_manual(values = c("red", "blue"))+
  scale_color_manual(values = c("red", "blue"))+
  theme_bw()+theme(panel.border = element_blank(),
                   panel.grid.major = element_blank(),
                   panel.grid.minor = element_blank(),
                   axis.line = element_line(colour = "black",size=1),
                   axis.ticks = element_line(colour = "black",size=1),
                   axis.text = element_text(size = 15,colour = "black"),
                   axis.title = element_text(size = 20),
                   plot.title = element_text(size = 20,hjust = 0.5),
                   legend.position="top")+
  labs(title = "Validation cohort", x = 'Model', y = "Probability")

raincloud_data<-read.xlsx("E:/project/pNR_rectalcancer/attention_radiomics/whole.xlsx",sheet = 'test_raincloud',startRow = 1,colNames = TRUE,rowNames = FALSE)
raincloud_data$label<-factor(raincloud_data$label)
ggplot(raincloud_data, aes(x = model, y = probability, fill = label)) +
  geom_flat_violin(aes(fill = label),position = position_nudge(x = .18, y = 0), adjust = 1.0, trim = TRUE, alpha = 1, colour = NA)+
  geom_point(aes(x = model, y = probability, colour = label),position = position_jitterdodge(jitter.width = 0.2, jitter.height = 0, dodge.width = 0.25), size = 2, shape = 20)+
  geom_boxplot(aes(x = model, y = probability, fill = label),position = position_dodge(width = 0.25), outlier.shape = NA, alpha = 1, width = .2, colour = "black")+
  # scale_colour_brewer(palette = "Set1")+
  # scale_fill_brewer(palette = "Set1")+
  scale_fill_manual(values = c("red", "blue"))+
  scale_color_manual(values = c("red", "blue"))+
  theme_bw()+theme(panel.border = element_blank(),
                   panel.grid.major = element_blank(),
                   panel.grid.minor = element_blank(),
                   axis.line = element_line(colour = "black",size=1),
                   axis.ticks = element_line(colour = "black",size=1),
                   axis.text = element_text(size = 15,colour = "black"),
                   axis.title = element_text(size = 20),
                   plot.title = element_text(size = 20,hjust = 0.5),
                   legend.position="top")+
  labs(title = "Testing cohort", x = 'Model', y = "Probability")


#
plot(CFDL_train,col='red',lwd=3,lty = 1, bty = "l", main="Training cohort",print.thres=F,legacy.axes=TRUE,cex.axis=1.4,cex.lab=1.7,cex.main=2) #
plot.roc(DL_train,add=TRUE,col='green',lty = 1,lwd=3) #F0E442
plot.roc(CFRS_train,add=TRUE,col='blue',lty = 1,lwd=3) #F0E442
plot.roc(RS_train,add=TRUE,col='orange',lty = 1,lwd=3) #F0E442
legend("bottomright",cex=1,bty="n",legend=c(paste("CFDL: ",signif(CFDL_train$auc,3)),
                                            paste("DL: ",signif(DL_train$auc,3)),
                                            paste("CFRS: ",signif(CFRS_train$auc,3)),
                                            paste("RS: ",signif(RS_train$auc,3))),
       col=c("red","green","blue","orange"),lty = c(1,1,1,1),lwd=3)

plot(CFDL_validation,col='red',lwd=3,lty = 1, main="Validation cohort",print.thres=F,legacy.axes=TRUE,cex.axis=1.4,cex.lab=1.7,cex.main=2) #
plot.roc(DL_validation,add=TRUE,col='green',lty = 1,lwd=3) #F0E442
plot.roc(CFRS_validation,add=TRUE,col='blue',lty = 1,lwd=3) #F0E442
plot.roc(RS_validation,add=TRUE,col='orange',lty = 1,lwd=3) #F0E442
legend("bottomright",cex=1,bty="n",legend=c(paste("CFDL: ",signif(CFDL_validation$auc,3)),
                                            paste("DL: ",signif(DL_validation$auc,3)),
                                            paste("CFRS: ",signif(CFRS_validation$auc,3)),
                                            paste("RS: ",signif(RS_validation$auc,3))),
       col=c("red","green","blue","orange"),lty = c(1,1,1,1),lwd=3)

plot(CFDL_test,col='red',lwd=3,lty = 1, main="Testing cohort",print.thres=F,legacy.axes=TRUE,cex.axis=1.4,cex.lab=1.7,cex.main=2) #local maximas main="radiomics",
plot.roc(DL_test,add=TRUE,col='green',lty = 1,lwd=3) #F0E442
plot.roc(CFRS_test,add=TRUE,col='blue',lty = 1,lwd=3) #F0E442
plot.roc(RS_test,add=TRUE,col='orange',lty = 1,lwd=3) #F0E442
legend("bottomright",cex=1,bty="n",legend=c(paste("CFDL: ",signif(CFDL_test$auc,3)),
                                            paste("DL: ",signif(DL_test$auc,3)),
                                            paste("CFRS: ",signif(CFRS_test$auc,3)),
                                            paste("RS: ",signif(RS_test$auc,3))),
       col=c("red","green","blue","orange"),lty = c(1,1,1,1),lwd=3)
###################################### Delong test
CFDL_DL<-roc.test(CFDL_train,DL_train)
CFDL_CFRS<-roc.test(CFDL_train,CFRS_train)
CFDL_RS<-roc.test(CFDL_train,RS_train)
DL_CFRS<-roc.test(DL_train,CFRS_train)
DL_RS<-roc.test(DL_train,RS_train)
CFRS_RS<-roc.test(CFRS_train,RS_train)

CFDL_DL<-roc.test(CFDL_validation,DL_validation)
CFDL_CFRS<-roc.test(CFDL_validation,CFRS_validation)
CFDL_RS<-roc.test(CFDL_validation,RS_validation)
DL_CFRS<-roc.test(DL_validation,CFRS_validation)
DL_RS<-roc.test(DL_validation,RS_validation)
CFRS_RS<-roc.test(CFRS_validation,RS_validation)

CFDL_DL<-roc.test(CFDL_test,DL_test)
CFDL_CFRS<-roc.test(CFDL_test,CFRS_test)
CFDL_RS<-roc.test(CFDL_test,RS_test)
DL_CFRS<-roc.test(DL_test,CFRS_test)
DL_RS<-roc.test(DL_test,RS_test)
CFRS_RS<-roc.test(CFRS_test,RS_test)

Cmatrix<-matrix(nrow = 4, ncol = 4)
Cmatrix[1,1]<-1
Cmatrix[1,2]<-CFDL_DL$p.value
Cmatrix[1,3]<-CFDL_CFRS$p.value
Cmatrix[1,4]<-CFDL_RS$p.value
Cmatrix[2,1]<-CFDL_DL$p.value
Cmatrix[2,2]<-1
Cmatrix[2,3]<-DL_CFRS$p.value
Cmatrix[2,4]<-DL_RS$p.value
Cmatrix[3,1]<-CFDL_CFRS$p.value
Cmatrix[3,2]<-DL_CFRS$p.value
Cmatrix[3,3]<-1
Cmatrix[3,4]<-CFRS_RS$p.value
Cmatrix[4,1]<-CFDL_RS$p.value
Cmatrix[4,2]<-DL_RS$p.value
Cmatrix[4,3]<-CFRS_RS$p.value
Cmatrix[4,4]<-1
colnames(Cmatrix)<-c('CFDL','DL','CFRS','RS')
rownames(Cmatrix)<-c('CFDL','DL','CFRS','RS')

library(corrplot)
corrplot(
  # 相关系数矩阵
  corr = Cmatrix,
  is.corr = FALSE,
  diag = TRUE,
  # cl.lim = c(0,1),
  order = 'original',
  method = 'circle',
  type = 'lower',
  addCoef.col = TRUE,
  tl.pos = NULL,
  tl.cex = 1,
  tl.col = "black",
  tl.srt = 0,
  tl.offset = 1,
  cl.pos = "r",
  number.cex = 1)

######################################multiple factor analysis
data_plot<-rbind.data.frame(train_clinical_all,validation_clinical_all)
data_plot<-rbind.data.frame(data_plot,test_clinical_all)
data_plot$set<-c(rep("train", nrow(train_clinical_all)),
                      rep("validation", nrow(validation_clinical_all)),
                      rep("test", nrow(test_clinical_all)))
					  
data_plot$sex<-factor(data_plot$sex,ordered = FALSE)

data_plot$T<-factor(data_plot$T,ordered = TRUE)

data_plot$N<-factor(data_plot$N,ordered = TRUE)

data_plot$CEA<-factor(data_plot$CEA,ordered = FALSE)

da=datadist(data_plot)
options(datadist="da")

f1.train <- lrm(label~age+sex+T+N+CFDL+DL+CFRS+RS+CEA, data=subset(data_plot, set=="train"), x=T,y=T,linear.predictors=T)
summary(f1.train)
coef(f1.train)

f1.train <- lrm(label~age+sex+T+N+CFDL+DL+CFRS+RS+CEA, data=subset(data_plot, set=="validation"), x=T,y=T,linear.predictors=T)
summary(f1.train)
coef(f1.train)

f1.train <- lrm(label~age+sex+T+N+CFDL+DL+CFRS+RS+CEA, data=subset(data_plot, set=="test"), x=T,y=T,linear.predictors=T)
summary(f1.train)
coef(f1.train)

library("forestplot")
data_forest <- read.xlsx("E:/project/pNR_rectalcancer/attention_radiomics/whole.xlsx",sheet = 'forest',startRow = 1,colNames = TRUE,rowNames = FALSE)
head(data_forest)
tabletext <- cbind(c("\nVariable",NA,data_forest$Variable,NA),
                   c("\nOdds Ratio (95% CI)",NA,ifelse(!is.na(data_forest$OR), paste(format(data_forest$OR,nsmall=2)," (",format(data_forest$Low,nsmall = 2)," ~ ",format(data_forest$High,nsmall = 2),")",sep=""), NA),NA),
                   c("\nP",NA,data_forest$P,NA))

head(tabletext)
forestplot(labeltext=tabletext,mean=c(NA,1,data_forest$OR,NA),
           lower=c(NA,NA,data_forest$Low,NA),upper=c(NA,NA,data_forest$High,NA),
           clip=c(0.01,100),
           zero = 1, 
           graphwidth = unit(.2,"npc"),
           graph.pos = 3,
           xlog=TRUE, 
           fn.ci_norm = fpDrawCircleCI, 
           boxsize = 0.3, 
           col=fpColors(line = "darkgrey", 
                        box="red"), 
           lty.ci = 7,   
           lwd.ci = 3,   
           ci.vertices.height = 0.15, 
           xticks = c(0.01, 1, 10, 100),
           hrzl_lines=list("3" = gpar(lwd=2, col="black"),
                           #"4" = gpar(lwd=60,lineend="butt", columns=c(1:4), col="#99999922"),
                           "39" = gpar(lwd=2, col="black")),
           txt_gp = fpTxtGp(ticks = gpar(cex = 0.5), xlab = gpar(cex = 0.7), cex = 0.7), 
           lineheight = "auto")
