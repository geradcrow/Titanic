library(data.table)
library(tidyverse)
library(caret)
library(skimr)
library(stringr)
library(ranger)
library(xgboost)

train_import <- fread("C://Users//gcrow//Desktop//Titanic//train.csv")
test_import <- fread("C://Users//gcrow//Desktop//Titanic//test.csv")

# Append test and train for data processing
train_import$RowType <- "train"
test_import$Survived <- NA
test_import$RowType <- "test"
 
 
 import_all <- bind_rows(train_import, test_import)
 
 #Calculate average age per gender and class for missing ages
 avg_age <- import_all %>%
   filter(is.na(Age) == FALSE) %>%
   group_by(Pclass, Sex) %>%  
   summarise(n=n(),AvgAge = mean(Age))
 
                        
 #Build features
prepped_data <- import_all %>% 
    left_join(avg_age,by = c('Pclass','Sex')) %>%
    mutate(Embarked = recode(Embarked, 'S' = "Southhampton", 'Q' = "Queenstown", 'C' = "Cherbourg"),
           ChildInd = case_when(Age <= 16 ~ 1, TRUE ~ 0),
           Class = as.factor(Pclass),
           Age = coalesce(Age,AvgAge),
           Relatives = SibSp + Parch,
           Deck = as.factor(str_sub(Cabin,1,1)),
           Survived = as.factor(Survived)) %>%
    separate(Name,c('Surname','Title'), sep = "[,.]") %>%
    mutate(AgeGroup = cut(Age, seq(0, 100, 20), labels = seq(5, 100, 20)),
           TravelledAlone = case_when(Relatives > 0 ~ 'No', TRUE ~ 'Yes'),
          TitleGroup = as.factor(case_when((str_trim(Title)) %in% c("Mr","Miss","Mrs","Master","Dr") ~ str_trim(Title), TRUE ~ "Other")))
  

#Split train and test
train_data <- prepped_data %>% filter(RowType == "train")

test_data <- prepped_data %>% filter(RowType == "test")


#Logistic regression model build


LogReg = train(form = Survived ~ Class  + TitleGroup + Age + Relatives + Deck + Sex + ChildInd,
                    data = train_data,
                    trControl = trainControl(method = "cv", number = 10),
                    method = "glm",
                    family = "binomial"
)


LogReg
summary(LogReg)
varImp(LogReg)

#LogReg$control
#LogReg$resample


#Apply model to test data

test_data$SurvivedLogReg <- predict(LogReg,newdata = test_data)


#Logistic regression submission

log_reg_output <- test_data %>% 
  select(PassengerId,Survived = SurvivedLogReg)
  
fwrite(log_reg_output, file = "C://Users//gcrow//Desktop//Titanic//titanic_output_logreg.csv")

str(log_reg_output)



#Random forest model build 
RF = train(form = Survived ~ Class  + TitleGroup + Age + Relatives + Deck + Sex + ChildInd, 
                data = train_data,
                trControl = trainControl(method = "cv", number = 10),
                method = "ranger"
)


RF1
RF
summary(RF1)
varImp(RF1)

plot(RF)
RF$finalModel
#Apply model to training data

test_data$SurvivedRF <- predict(RF,newdata = test_data)


#RF submission

RF_output <- test_data %>% 
  select(PassengerId,Survived = SurvivedRF)

fwrite(RF_output, file = "C://Users//gcrow//Desktop//Titanic//titanic_output_RF.csv")

str(RF_output)



#XGBoost model
XGB = train(form = Survived ~ Class  + TitleGroup + Age + Relatives + Deck + Sex + ChildInd, 
           data = train_data,
           trControl = trainControl(method = "cv", number = 10),
           method = "xgbTree"
)


XGB
summary(XGB)
varImp(XGB)
XGB$finalModel

plot(XGB)


#Apply model to training data

test_data$SurvivedXGB <- predict(XGB,newdata = test_data)


#Submission

XGB_output <- test_data %>% 
  select(PassengerId,Survived = SurvivedXGB)

fwrite(RF_output, file = "C://Users//gcrow//Desktop//Titanic//titanic_output_XGB.csv")



