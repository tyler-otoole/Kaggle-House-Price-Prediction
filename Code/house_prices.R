library(tidyverse)
library(ggcorrplot)
library(scales)
library(ggrepel)
library(plyr)
library(kableExtra)
library(ggjoy)
library(randomForest)
library(gridExtra)
library(caret)
library(Rmisc)
library(psych)
library(corrplot)
library(xgboost)

setwd("C:/Users/tyler/OneDrive/Documents/R Files/Kaggle House Price Prediction Competition")

#######
# EDA
######


train <- read.csv("train.csv", stringsAsFactors = F)
test <- read.csv("test.csv", stringsAsFactors = F)

glimpse(train) # 1460 x 80 (not including "Id" column)
glimpse(test) # 1459 X 79 (not including "Id" column)

# need the test labesl to submit results for competition...

test_labels <- test$Id
train_labels <- train$Id

test$Id <- NULL
train$Id <- NULL

# Test has no sale price variable; this is what we're trying to predict.
# Adding Sale price column but NA values. To merge data.

test$SalePrice <- NA

full_data <- rbind(train, test)

glimpse(full_data) # 2919 x 80

# Sale Price

ggplot(full_data[!is.na(full_data$SalePrice),], aes(x=SalePrice)) +
  geom_histogram(binwidth = 10000) +
  scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  xlab("") + 
  ylab("") + 
  ggtitle("Histogram of Sale Price") +
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5))

summary(full_data$SalePrice) # Median = 163,000

numericVars <- which(sapply(full_data, is.numeric)) # All numeric vars
numericVarNames <- names(numericVars) # Names of numeric vars

full_numVar <- full_data[, numericVars]

cor_numVar <- cor(full_numVar, use="pairwise.complete.obs") #correlations

# Get only highly correlated variables
# Correlation Matrix

cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

ggcorrplot(cor_numVar, hc.order = TRUE, type = "lower",
           outline.col = "white", lab = TRUE,
           ggtheme = ggplot2::theme_minimal,
           colors = c("#6D9EC1", "white", "#E46726")) +
           xlab("") + 
           ylab("") + 
           ggtitle("Correlation Matrix of Highly Correlated Variables")

# Highest 3: OverallQual, GrLivArea, GarageCars
# Issues with multico - GarageCars highly correlated with GarageArea, for example.

# OverallQual

options(scipen=999)
ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(OverallQual), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='Overall Quality', y = 'Sale Price') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Sale Price and Overall Quality") + 
  ylab("") +
  xlab("") +
  theme(plot.title = element_text(hjust = .5))

ggplot(full_data[!is.na(full_data$SalePrice),], aes(x=OverallQual, y=SalePrice)) +
  geom_jitter(aes(alpha = .5)) + 
  stat_smooth(color = "DarkBlue") + 
  theme_minimal() +
  ggtitle("Sale Price and Overall Quality") + 
  scale_y_continuous(labels = comma) + 
  ylab("") +
  xlab("") +
  theme(plot.title = element_text(hjust = .5)) +
  theme(legend.position = "none")
  
# Above ground living area

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=GrLivArea, y=SalePrice))+
  geom_point() + 
  geom_smooth(method = "lm", color="Blue") +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_text_repel(aes(label = ifelse(full_data$GrLivArea[!is.na(full_data$SalePrice)]>4500, 
                                     rownames(full_data), ''))) +
  theme_minimal() +
  ggtitle("Above Ground Living Area and Sale Price") +
  ylab("") +
  xlab("Living Area") +
  theme(plot.title = element_text(hjust = .5))

full_data[c(524, 1299), c('SalePrice', 'GrLivArea', 'OverallQual')] # Examining potential outliers

# GarageCars 

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(GarageCars), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Garage Car Capacity and Sale Price") +
  ylab("") +
  xlab("") +
  theme(plot.title = element_text(hjust = .5))

# Look at potential outliers

full_data[full_data$GarageCars==3 & full_data$SalePrice>700000 & !is.na(full_data$SalePrice),
          c('GarageCars',  'OverallQual')]

#############################################
# Missing data. Label encoding. Factorizing
#############################################

NAcol <- which(colSums(is.na(full_data)) > 0)
sort(colSums(sapply(full_data[NAcol], is.na)), decreasing = TRUE)

# Pool Quality has the highest number of NAs. This is expected. Pools aren't popular in Iowa.

# Ex   Excellent
# Gd   Good
# TA   Average/Typical
# Fa   Fair
# NA   No Pool

full_data$PoolQC[is.na(full_data$PoolQC)] <- 'none' # Change NA's to "none"

# Ordinal

Qualities <- c('none' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)

full_data$PoolQC<-as.integer(revalue(full_data$PoolQC, Qualities))
table(full_data$PoolQC)

full_data[full_data$PoolArea>0 & full_data$PoolQC==0, c('PoolArea', 'PoolQC', 
                                                        'OverallQual', 'ExterQual',
                                                        'ExterCond')]

# 2421, 2504, 2600 have pool area, but no pool quality.
# Impute based on ExterCond becasue pools are outside. 
# All are TA or Average

full_data$PoolQC[2421] <- 3
full_data$PoolQC[2504] <- 3
full_data$PoolQC[2600] <- 3

#2 MiscFeature 

# Elev Elevator
# Gar2 2nd Garage (if not described in garage section)
# Othr Other
# Shed Shed (over 100 SF)
# TenC Tennis Court
# NA   None

full_data$MiscFeature[is.na(full_data$MiscFeature)] <- 'none'
full_data$MiscFeature <- as.factor(full_data$MiscFeature)

table(full_data$MiscFeature)

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=MiscFeature, y=SalePrice))+
  geom_bar(stat='summary', fun.y = "median", fill='#6D9EC1') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Features and Sale Price") +
  ylab("") +
  xlab("") +
  theme(plot.title = element_text(hjust = .5)) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

#3 Alley (Type of alley access to property)

# Grvl Gravel
# Pave Paved
# NA   No alley access

full_data$Alley[is.na(full_data$Alley)] <- 'none'
full_data$Alley <- as.factor(full_data$Alley)

table(full_data$Alley)

#4 Fence.

# GdPrv Good Privacy
# MnPrv Minimum Privacy
# GdWo Good Wood
# MnWw Minimum Wood/Wire
# NA No Fence

full_data$Fence[is.na(full_data$Fence)] <- 'none'
table(full_data$Fence)

ddply(full_data[!is.na(full_data$SalePrice),],~Fence,summarise,median=median(SalePrice))

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(Fence), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Fence Types vs. Sale Price") +
  theme(plot.title = element_text(hjust = 0.5))

full_data$Fence <- as.factor(full_data$Fence)

#5 FirePlaceQu

# Ex   Excellent - Exceptional Masonry Fireplace
# Gd   Good - Masonry Fireplace in main level
# TA   Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
# Fa   Fair - Prefabricated Fireplace in basement
# Po   Poor - Ben Franklin Stove
# NA   No Fireplace

full_data$FireplaceQu[is.na(full_data$FireplaceQu)] <- 'none'
full_data$FireplaceQu<-as.integer(revalue(full_data$FireplaceQu, Qualities))
table(full_data$FireplaceQu)

#6 LotFrontage: Linear feet of street connected to property

ggplot(full_data[!is.na(full_data$LotFrontage),], aes(x=reorder(as.factor(Neighborhood), LotFrontage), y=LotFrontage)) +
  geom_bar(stat='summary', fun.y = "median", fill = "#6D9EC1") +
  theme(axis.text.x = element_text(hjust = 1)) +
  coord_flip() +
  xlab("") +
  ylab("") + 
  ggtitle("Median Lot Frontage by Neighborhood")
  theme_minimal()

# Impute with median of neighborhood 

for (i in 1:nrow(full_data)){
  if(is.na(full_data$LotFrontage[i])){
    full_data$LotFrontage[i] <- as.integer(median(full_data$LotFrontage[full_data$Neighborhood==full_data$Neighborhood[i]], na.rm=TRUE)) 
  }
}

# Lot Shape

# Reg  Regular 
# IR1  Slightly irregular
# IR2  Moderately Irregular
# IR3  Irregular

# Ordinal

full_data$LotShape<-as.integer(revalue(full_data$LotShape, c('IR3'=0, 'IR2'=1, 'IR1'=2, 'Reg'=3)))
table(full_data$LotShape)

# Lot configuration.

# Inside   Inside lot
# Corner   Corner lot
# CulDSac  Cul-de-sac
# FR2  Frontage on 2 sides of property
# FR3  Frontage on 3 sides of property

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(LotConfig), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Lot Configuration vs. Sale Price") +
  theme(plot.title = element_text(hjust = 0.5))

# Factor

full_data$LotConfig <- as.factor(full_data$LotConfig)
table(full_data$LotConfig)

#7 GarageYrBlt

# Replace missings with year built

full_data$GarageYrBlt[is.na(full_data$GarageYrBlt)] <- full_data$YearBuilt[is.na(full_data$GarageYrBlt)]

table(full_data$GarageYrBlt)

# Changing 2207 year to 2007

full_data <- full_data %>% mutate(GarageYrBlt = replace(GarageYrBlt, GarageYrBlt == 2207, 2007))

full_data[!is.na(full_data$GarageType) & is.na(full_data$GarageFinish), c('GarageCars', 'GarageArea', 
                                                                          'GarageType', 'GarageCond', 
                                                                          'GarageQual', 'GarageFinish',
                                                                          'OverallQual', 'ExterQual')]
# Using exterior quality to impute garage condition and garage quality
# observation 2577 does not have a garage.

full_data$GarageCond[2127] <- 'TA'
full_data$GarageQual[2127] <- 'TA'
full_data$GarageFinish[2127] <- 'Unf'

full_data$GarageCars[2577] <- 0
full_data$GarageArea[2577] <- 0
full_data$GarageType[2577] <- NA

# Garage type: Garage Location

# 2Types   More than one type of garage
# Attchd   Attached to home
# Basment  Basement Garage
# BuiltIn  Built-In (Garage part of house - typically has room above garage)
# CarPort  Car Port
# Detchd   Detached from home
# NA   No Garage

# Factor

full_data$GarageType[is.na(full_data$GarageType)] <- 'No Garage'
full_data$GarageType <- as.factor(full_data$GarageType)
table(full_data$GarageType)

# Garage Finish: Interior finish of the garage

# Fin  Finished
# RFn  Rough Finished  
# Unf  Unfinished
# NA   No Garage     

# Ordinal

full_data$GarageFinish[is.na(full_data$GarageFinish)] <- 'none'
Finish <- c('none'=0, 'Unf'=1, 'RFn'=2, 'Fin'=3)

full_data$GarageFinish<-as.integer(revalue(full_data$GarageFinish, Finish))
table(full_data$GarageFinish)

# GarageQual

# Ex   Excellent
# Gd   Good
# TA   Typical/Average
# Fa   Fair
# Po   Poor
# NA   No Garage

# Ordinal

full_data$GarageQual[is.na(full_data$GarageQual)] <- 'none'
full_data$GarageQual<-as.integer(revalue(full_data$GarageQual, Qualities))
table(full_data$GarageQual)

# GarageCond
# Ordinal

full_data$GarageCond[is.na(full_data$GarageCond)] <- 'none'
full_data$GarageCond<-as.integer(revalue(full_data$GarageCond, Qualities))
table(full_data$GarageCond)

#8 Basement variables

full_data[!is.na(full_data$BsmtFinType1) & 
      (is.na(full_data$BsmtCond)|is.na(full_data$BsmtQual)
                                |is.na(full_data$BsmtExposure)|is.na(full_data$BsmtFinType2)), c('BsmtQual', 
                                                                                     'BsmtCond', 
                                                                                     'BsmtExposure', 
                                                                                     'BsmtFinType1', 
                                                                                     'BsmtFinType2')]

# Imputing modes for missing values

full_data$BsmtFinType2[333] <- names(sort(-table(full_data$BsmtFinType2)))[1]
full_data$BsmtExposure[c(949, 1488, 2349)] <- names(sort(-table(full_data$BsmtExposure)))[1]
full_data$BsmtCond[c(2041, 2186, 2525)] <- names(sort(-table(full_data$BsmtCond)))[1]
full_data$BsmtQual[c(2218, 2219)] <- names(sort(-table(full_data$BsmtQual)))[1]

# BasementQual: Height of basement

# Ex   Excellent (100+ inches) 
# Gd   Good (90-99 inches)
# TA   Typical (80-89 inches)
# Fa   Fair (70-79 inches)
# Po   Poor (<70 inches
# NA   No Basement

# Ordinal

full_data$BsmtQual[is.na(full_data$BsmtQual)] <- 'none'
full_data$BsmtQual<-as.integer(revalue(full_data$BsmtQual, Qualities))
table(full_data$BsmtQual)

# BsmtCond: Condition of basement

# Ex   Excellent
# Gd   Good
# TA   Typical - slight dampness allowed
# Fa   Fair - dampness or some cracking or settling
# Po   Poor - Severe cracking, settling, or wetness
# NA   No Basement

full_data$BsmtCond[is.na(full_data$BsmtCond)] <- 'none'
full_data$BsmtCond<-as.integer(revalue(full_data$BsmtCond, Qualities))
table(full_data$BsmtCond)

# BsmtExposure: Walkout or garden level walls

# Gd   Good Exposure
# Av   Average Exposure (split levels or foyers typically score average or above)  
# Mn   Mimimum Exposure
# No   No Exposure
# NA   No Basement

# Ordinal

full_data$BsmtExposure[is.na(full_data$BsmtExposure)] <- 'none'
Exposure <- c('none'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)

full_data$BsmtExposure<-as.integer(revalue(full_data$BsmtExposure, Exposure))
table(full_data$BsmtExposure)

# BsmtFinType1: Rating of basement finished area (Ordinal)

# GLQ  Good Living Quarters
# ALQ  Average Living Quarters
# BLQ  Below Average Living Quarters   
# Rec  Average Rec Room
# LwQ  Low Quality
# Unf  Unfinshed
# NA   No Basement

full_data$BsmtFinType1[is.na(full_data$BsmtFinType1)] <- 'none'
FinType <- c('none'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)

full_data$BsmtFinType1<-as.integer(revalue(full_data$BsmtFinType1, FinType))
table(full_data$BsmtFinType1)

# BsmtFinType2: Rating of basement finished area (if multiple types)

# GLQ  Good Living Quarters
# ALQ  Average Living Quarters
# BLQ  Below Average Living Quarters   
# Rec  Average Rec Room
# LwQ  Low Quality
# Unf  Unfinshed
# NA   No Basement

# Make ordinal with fintype vector

full_data$BsmtFinType2[is.na(full_data$BsmtFinType2)] <- 'none'
FinType <- c('none'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)

full_data$BsmtFinType2<-as.integer(revalue(full_data$BsmtFinType2, FinType))
table(full_data$BsmtFinType2)

# Remaining basement vars

full_data[(is.na(full_data$BsmtFullBath)|is.na(full_data$BsmtHalfBath)
        |is.na(full_data$BsmtFinSF1)
        |is.na(full_data$BsmtFinSF2)
        |is.na(full_data$BsmtUnfSF)
        |is.na(full_data$TotalBsmtSF)), c('BsmtQual', 'BsmtFullBath', 
                                          'BsmtHalfBath', 'BsmtFinSF1', 
                                          'BsmtFinSF2', 'BsmtUnfSF', 
                                          'TotalBsmtSF')]

# BsmtFullBath: Basement full bathrooms

full_data$BsmtFullBath[is.na(full_data$BsmtFullBath)] <-0
table(full_data$BsmtFullBath)

# BsmtHalfBath: Basement half bathrooms

full_data$BsmtHalfBath[is.na(full_data$BsmtHalfBath)] <-0
table(full_data$BsmtHalfBath)

# BsmtFinSF1: Type 1 finished square feet

full_data$BsmtFinSF1[is.na(full_data$BsmtFinSF1)] <-0

# BsmtFinSF2: Type 2 finished square feet

full_data$BsmtFinSF2[is.na(full_data$BsmtFinSF2)] <-0

# BsmtUnfSF: Unfinished square feet of basement area

full_data$BsmtUnfSF[is.na(full_data$BsmtUnfSF)] <-0

# TotalBsmtSF: Total square feet of basement area

full_data$TotalBsmtSF[is.na(full_data$TotalBsmtSF)] <-0

# Masonry Variables

# Masonry veneer type, and masonry veneer area
# Masonry veneer type has 24 NA, masonry veneer area has 23

full_data[is.na(full_data$MasVnrType) & !is.na(full_data$MasVnrArea), c('MasVnrType', 'MasVnrArea')]

# 2611 - imputing mode seems to be the right move

full_data$MasVnrType[2611] <- names(sort(-table(full_data$MasVnrType)))[2] #taking the 2nd value as the 1st is 'none'

# Type

# BrkCmn   Brick Common
# BrkFace  Brick Face
# CBlock   Cinder Block
# None     None
# Stone    Stone

full_data$MasVnrType[is.na(full_data$MasVnrType)] <- 'None'

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(MasVnrType), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Masonry Veneer Type vs. Price") +
  theme(plot.title = element_text(hjust = 0.5))

# Looking at boxplot... Variables look ordinal. I'm going to code them as none = 0
# brkcmn = 1, brkface = 2, stone = 3

Masonry <- c('None'=0, 'BrkCmn'=1, 'BrkFace'=2, 'Stone'=3)
full_data$MasVnrType<-as.integer(revalue(full_data$MasVnrType, Masonry))
table(full_data$MasVnrType)

# Veneer area

full_data$MasVnrArea[is.na(full_data$MasVnrArea)] <-0

# MSZoning: Identifies the general zoning classification of the sale

# A    Agriculture
# C    Commercial
# FV   Floating Village Residential
# I    Industrial
# RH   Residential High Density
# RL   Residential Low Density
# RP   Residential Low Density Park 
# RM   Residential Medium Density

full_data[is.na(full_data$MSZoning), c('MSZoning', 'OverallQual')] 

# 1916, 2217, 2251, 2905 missing zoning
# Let's look at the neighborhoods to try to determine what kind of zoning class.

full_data[1916, c('Neighborhood', 'MSZoning')] # IDOTRR
full_data[2217, c('Neighborhood', 'MSZoning')] # IDOTRR
full_data[2251, c('Neighborhood', 'MSZoning')] # IDOTRR
full_data[2905, c('Neighborhood', 'MSZoning')] # Mitchel

idotrr <- full_data %>% filter(Neighborhood == "IDOTRR")
names(sort(-table(idotrr$MSZoning)))[1] # Mode MSZoning for IDOTRR = RM, coding as such

mitchel <- full_data %>% filter(Neighborhood == "Mitchel")
names(sort(-table(mitchel$MSZoning)))[1] # Mode MSZoning for Michel = RL, coding as such


full_data$MSZoning[1916] <- "RM"
full_data$MSZoning[2217] <- "RM"
full_data$MSZoning[2251] <- "RM"
full_data$MSZoning[2905] <- "RL"

full_data$MSZoning <- as.factor(full_data$MSZoning)
table(full_data$MSZoning)

# Kitchen quality - 1 na; observation 1556

# Ex   Excellent
# Gd   Good
# TA   Typical/Average
# Fa   Fair
# Po   Poor

full_data[is.na(full_data$KitchenQual), c('KitchenQual', 'OverallQual')] # Overall Quality is a 5/10

# Recoding as Fair. Fair is about a 5/10 in my opinion
# Also recoding as ordinal using qualities vector

full_data$KitchenQual[1556] <- "Fa"
full_data$KitchenQual<-as.integer(revalue(full_data$KitchenQual, Qualities))
table(full_data$KitchenQual) # Strange that there are no poor quality kitchens?

# Utilities: Type of utilities available

# AllPub   All public Utilities (E,G,W,& S)    
# NoSewr   Electricity, Gas, and Water (Septic Tank)
# NoSeWa   Electricity and Gas Only
# ELO      Electricity only

table(full_data$Utilities)

# ? all besides 1 have all. This variable is useless. 

full_data$Utilities <- NULL

# Home functionality (ordinal)

# Typ  Typical Functionality
# Min1 Minor Deductions 1
# Min2 Minor Deductions 2
# Mod  Moderate Deductions
# Maj1 Major Deductions 1
# Maj2 Major Deductions 2
# Sev  Severely Damaged
# Sal  Salvage only

full_data[is.na(full_data$Functional), c('Functional', 'OverallQual')] # 2217 overallqual is 1
                                                                       # 2472 overallqual is 4

# Going to ding 2217 a bit, give them a Min1
# Will give 2472 a Typ

full_data$Functional[2217] <- "Min1"
full_data$Functional[2474] <- "Typ"

# Encoding as integer (ordinal)

full_data$Functional <- as.integer(revalue(full_data$Functional, c('Sal'=0,'Sev'=1, 'Maj2'=2, 
                                                                   'Maj1'=3, 'Mod'=4, 
                                                                   'Min2'=5, 'Min1'=6, 'Typ'=7)))

table(full_data$Functional)

# Exterior1st: Exterior covering on house

# AsbShng  Asbestos Shingles
# AsphShn  Asphalt Shingles
# BrkComm  Brick Common
# BrkFace  Brick Face
# CBlock   Cinder Block
# CemntBd  Cement Board
# HdBoard  Hard Board
# ImStucc  Imitation Stucco
# MetalSd  Metal Siding
# Other    Other
# Plywood  Plywood
# PreCast  PreCast 
# Stone    Stone
# Stucco   Stucco
# VinylSd  Vinyl Siding
# Wd Sdng  Wood Siding
# WdShing  Wood Shingles

full_data[is.na(full_data$Exterior1st), c('Exterior1st', 'Neighborhood', 'OverallQual', 'YearBuilt')]

# observation 2152: neighborhood is Edwards, Overall Quality is 5, Year Built is 1940

extna <- full_data %>% filter(Neighborhood == 'Edwards' & YearBuilt == 1940)
names(sort(-table(extna$Exterior1st)))[1] 

# Metal Siding (MetalSd) is the most common value for Exterior1st in 
# this neighborhood, for houses built in 1940. Will impute as such.

full_data$Exterior1st[2152] <- "MetalSd"

full_data$Exterior1st <- as.factor(full_data$Exterior1st) # Factor
table(full_data$Exterior1st)

# Exterior2nd: Exterior covering on house (if more than one material)

# AsbShng  Asbestos Shingles
# AsphShn  Asphalt Shingles
# BrkComm  Brick Common
# BrkFace  Brick Face
# CBlock   Cinder Block
# CemntBd  Cement Board
# HdBoard  Hard Board
# ImStucc  Imitation Stucco
# MetalSd  Metal Siding
# Other    Other
# Plywood  Plywood
# PreCast  PreCast
# Stone    Stone
# Stucco   Stucco
# VinylSd  Vinyl Siding
# Wd Sdng  Wood Siding
# WdShing  Wood Shingles

full_data[is.na(full_data$Exterior2nd), c('Exterior2nd', 'Neighborhood', 'OverallQual', 'YearBuilt')]

# observation 2152 again

extna2 <- full_data %>% filter(Neighborhood == 'Edwards' & YearBuilt == 1940)
names(sort(-table(extna$Exterior2nd)))[1] 

# MetalSd is the mode for this neighborhood and year built

full_data$Exterior2nd[2152] <- "MetalSd"
full_data$Exterior2nd <- as.factor(full_data$Exterior2nd)
table(full_data$Exterior2nd)

#15 Electrical System

# SBrkr    Standard Circuit Breakers & Romex
# FuseA    Fuse Box over 60 AMP and all Romex wiring (Average) 
# FuseF    60 AMP Fuse Box and mostly Romex wiring (Fair)
# FuseP    60 AMP Fuse Box and mostly knob & tube wiring (poor)
# Mix      Mixed

full_data[is.na(full_data$Electrical), c('Electrical', 'Neighborhood', 'OverallQual', 'YearBuilt')]

# Observation 1380; Neighborhood is Timber, Year built is 2006. Lets look at similar homes to imput missing value

elecna <- full_data %>% filter(Neighborhood == 'Timber' & YearBuilt == 2006)
names(sort(-table(elecna$Electrical)))[1]

full_data$Electrical[1380] <- "SBrkr"
full_data$Electrical <- as.factor(full_data$Electrical)
table(full_data$Electrical)

#16 SaleType: Type of sale

# WD       Warranty Deed - Conventional
# CWD      Warranty Deed - Cash
# VWD      Warranty Deed - VA Loan
# New      Home just constructed and sold
# COD      Court Officer Deed/Estate
# Con      Contract 15% Down payment regular terms
# ConLw    Contract Low Down payment and low interest
# ConLI    Contract Low Interest
# ConLD    Contract Low Down
# Oth      Other

# Just going to impute the mode for this bad boy

full_data$SaleType[is.na(full_data$SaleType)] <- names(sort(-table(full_data$SaleType)))[1]

full_data$SaleType <- as.factor(full_data$SaleType)
table(full_data$SaleType)

# SaleCondition: Condition of sale

# Normal   Normal Sale
# Abnorml  Abnormal Sale -  trade, foreclosure, short sale
# AdjLand  Adjoining Land Purchase
# Alloca   Allocation - two linked properties with separate deeds, typically condo with a garage unit  
# Family   Sale between family members
# Partial  Home was not completed when last assessed (associated with New Homes)

full_data$SaleCondition <- as.factor(full_data$SaleCondition)
table(full_data$SaleCondition)

# Now sale price is the only variable with NAs. 1459 of them. This corresponds to the test set. 
# We GROOVY

#########################################
# Examine remaining Character variables
#########################################

# Which columns are left?

character_vector <- names(full_data[,sapply(full_data, is.character)])
character_vector

# Exterior variables

full_data$ExterQual<-as.integer(revalue(full_data$ExterQual, Qualities))
full_data$ExterCond<-as.integer(revalue(full_data$ExterCond, Qualities))

# Foundation: Type of foundation

# BrkTil          Brick & Tile
# CBlock          Cinder Block
# PConc           Poured Contrete 
# Slab            Slab
# Stone           Stone
# Wood            Wood

# Factor

full_data$Foundation <- as.factor(full_data$Foundation)

# Heating and Airconditioning Variables
# Heating: Type of heating

# Floor    Floor Furnace
# GasA Gas forced warm air furnace
# GasW Gas hot water or steam heat
# Grav Gravity furnace 
# OthW Hot water or steam heat other than gas
# Wall Wall furnace

full_data$Heating <- as.factor(full_data$Heating)

# HeatingQC: Heating quality and condition

# Ex   Excellent
# Gd   Good
# TA   Average/Typical
# Fa   Fair
# Po   Poor

# Ordinal

full_data$HeatingQC<-as.integer(revalue(full_data$HeatingQC, Qualities))

# CentralAir: Central air conditioning

full_data$CentralAir<-as.integer(revalue(full_data$CentralAir, c('N'=0, 'Y'=1)))
table(full_data$CentralAir)

# Roof variables
# RoofStyle: Type of roof

# Flat Flat
# Gable    Gable
# Gambrel  Gabrel (Barn)
# Hip  Hip
# Mansard  Mansard
# Shed Shed

full_data$RoofStyle <- as.factor(full_data$RoofStyle)
table(full_data$RoofStyle)

# RoofMatl: Roof material

# ClyTile  Clay or Tile
# CompShg  Standard (Composite) Shingle
# Membran  Membrane
# Metal    Metal
# Roll Roll
# Tar&Grv  Gravel & Tar
# WdShake  Wood Shakes
# WdShngl  Wood Shingles

full_data$RoofMatl <- as.factor(full_data$RoofMatl)
table(full_data$RoofMatl)

# Land Variables
# LandContour: Flatness of the property

# Lvl  Near Flat/Level 
# Bnk  Banked - Quick and significant rise from street grade to building
# HLS  Hillside - Significant slope from side to side
# Low  Depression

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(LandContour), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Flatness of Property and Selling Price") +
  theme(plot.title = element_text(hjust = 0.5))

# Factor

full_data$LandContour <- as.factor(full_data$LandContour)
table(full_data$LandContour)

# LandSlope: Slope of property

# Gtl  Gentle slope
# Mod  Moderate Slope  
# Sev  Severe Slope

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(LandSlope), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Slope of Property and Selling Price") +
  theme(plot.title = element_text(hjust = 0.5))

# Ordinal

full_data$LandSlope<-as.integer(revalue(full_data$LandSlope, c('Sev'=0, 'Mod'=1, 'Gtl'=2)))
table(full_data$LandSlope)

# Dwelling variables
# BldgType: Type of dwelling

# 1Fam Single-family Detached  
# 2FmCon   Two-family Conversion; originally built as one-family dwelling
# Duplx    Duplex
# TwnhsE   Townhouse End Unit
# TwnhsI   Townhouse Inside Unit

# Check ordinality

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(BldgType), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Type of Dwelling vs. Sale Price") +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(full_data[!is.na(full_data$SalePrice),], aes(x=as.factor(BldgType), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median")+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  theme_minimal() + 
  xlab("") + 
  ylab("") +
  ggtitle("Median Sale Price by Type of Dwelling") +
  theme(plot.title = element_text(hjust = 0.5))

# Factor

full_data$BldgType <- as.factor(full_data$BldgType)
table(full_data$BldgType)

# HouseStyle: Style of dwelling

# 1Story   One story
# 1.5Fin   One and one-half story: 2nd level finished
# 1.5Unf   One and one-half story: 2nd level unfinished
# 2Story   Two story
# 2.5Fin   Two and one-half story: 2nd level finished
# 2.5Unf   Two and one-half story: 2nd level unfinished
# SFoyer   Split Foyer
# SLvl Split Level


ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(HouseStyle), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("House Style and Sale Price") +
  theme(plot.title = element_text(hjust = 0.5))

# Factor

full_data$HouseStyle <- as.factor(full_data$HouseStyle)
table(full_data$HouseStyle)

# Neighborhood: Physical locations within Ames city limits

# Blmngtn  Bloomington Heights
# Blueste  Bluestem
# BrDale   Briardale
# BrkSide  Brookside
# ClearCr  Clear Creek
# CollgCr  College Creek
# Crawfor  Crawford
# Edwards  Edwards
# Gilbert  Gilbert
# IDOTRR   Iowa DOT and Rail Road
# MeadowV  Meadow Village
# Mitchel  Mitchell
# Names    North Ames
# NoRidge  Northridge
# NPkVill  Northpark Villa
# NridgHt  Northridge Heights
# NWAmes   Northwest Ames
# OldTown  Old Town
# SWISU    South & West of Iowa State University
# Sawyer   Sawyer
# SawyerW  Sawyer West
# Somerst  Somerset
# StoneBr  Stone Brook
# Timber   Timberland
# Veenker  Veenker

full_data$Neighborhood <- as.factor(full_data$Neighborhood)
table(full_data$Neighborhood)

# Condition1: Proximity to various conditions

# Artery   Adjacent to arterial street
# Feedr    Adjacent to feeder street   
# Norm Normal  
# RRNn Within 200' of North-South Railroad
# RRAn Adjacent to North-South Railroad
# PosN Near positive off-site feature--park, greenbelt, etc.
# PosA Adjacent to postive off-site feature
# RRNe Within 200' of East-West Railroad
# RRAe Adjacent to East-West Railroad

full_data$Condition1 <- as.factor(full_data$Condition1)
table(full_data$Condition1)

# Condition2: Proximity to various conditions (if more than one is present)

# Artery   Adjacent to arterial street
# Feedr    Adjacent to feeder street   
# Norm     Normal  
# RRNn     Within 200' of North-South Railroad
# RRAn     Adjacent to North-South Railroad
# PosN     Near positive off-site feature--park, greenbelt, etc.
# PosA     Adjacent to postive off-site feature
# RRNe     Within 200' of East-West Railroad
# RRAe     Adjacent to East-West Railroad

full_data$Condition2 <- as.factor(full_data$Condition2)
table(full_data$Condition2)

# Street: Type of road access to property

# Grvl Gravel  
# Pave Paved

# Ordinal

full_data$Street<-as.integer(revalue(full_data$Street, c('Grvl'=0, 'Pave'=1)))
table(full_data$Street)

# PavedDrive: Paved driveway

# Y    Paved 
# P    Partial Pavement
# N    Dirt/Gravel

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(PavedDrive), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Driveway Type and Selling Price") +
  theme(plot.title = element_text(hjust = 0.5))

# Ordinal

full_data$PavedDrive<-as.integer(revalue(full_data$PavedDrive, c('N'=0, 'P'=1, 'Y'=2)))
table(full_data$PavedDrive)

###########################
# Numerics to Factors
###########################

# 20  1-STORY 1946 & NEWER ALL STYLES
# 30  1-STORY 1945 & OLDER
# 40  1-STORY W/FINISHED ATTIC ALL AGES
# 45  1-1/2 STORY - UNFINISHED ALL AGES
# 50  1-1/2 STORY FINISHED ALL AGES
# 60  2-STORY 1946 & NEWER
# 70  2-STORY 1945 & OLDER
# 75  2-1/2 STORY ALL AGES
# 80  SPLIT OR MULTI-LEVEL
# 85  SPLIT FOYER
# 90  DUPLEX - ALL STYLES AND AGES
# 120  1-STORY PUD (Planned Unit Development) - 1946 & NEWER
# 150  1-1/2 STORY PUD - ALL AGES
# 160  2-STORY PUD - 1946 & NEWER
# 180  PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
# 190  2 FAMILY CONVERSION - ALL STYLES AND AGES

str(full_data$MSSubClass)
full_data$MSSubClass<-revalue(as.factor(full_data$MSSubClass), c('20'='1 story 1946+', 
                                                      '30'='1 story 1945-', 
                                                      '40'='1 story unf attic', 
                                                      '45'='1,5 story unf', 
                                                      '50'='1.5 story fin', 
                                                      '60'='2 story 1946+', 
                                                      '70'='2 story 1945-', 
                                                      '75'='2.5 story all ages', 
                                                      '80'='split/multi level', 
                                                      '85'='split foyer', 
                                                      '90'='duplex all style/age', 
                                                      '120'='1 story PUD 1946+', 
                                                      '150'='1.5 story PUD all', 
                                                      '160'='2 story PUD 1946+', 
                                                      '180'='PUD multilevel', 
                                                      '190'='2 family conversion'))


full_data$MoSold <- as.factor(full_data$MoSold)

# Keeping year as numeric to create an "Age" variable later...

ys <- ggplot(full_data[!is.na(full_data$SalePrice),], aes(x=as.factor(YrSold), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median")+
  scale_y_continuous(breaks= seq(0, 800000, by=25000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..)) +
  coord_cartesian(ylim = c(0, 200000)) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") + #dashed line is median SalePrice
  ggtitle("Year") +
  theme(plot.title = element_text(hjust = .5)) +
  theme_minimal() +
  ylab("") +
  xlab("")

ms <- ggplot(full_data[!is.na(full_data$SalePrice),], aes(x=MoSold, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median")+
  scale_y_continuous(breaks= seq(0, 800000, by=25000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..)) +
  coord_cartesian(ylim = c(0, 200000)) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") +#dashed line is median SalePrice
  ggtitle("Month") +
  theme(plot.title = element_text(hjust = .5)) +
  theme_minimal() +
  ylab("") +
  xlab("")

grid.arrange(ys, ms, widths=c(1,2))

# Correlation Plot

numericVars <- which(sapply(full_data, is.numeric)) #index vector numeric variables
factorVars <- which(sapply(full_data, is.factor)) #index vector factor variables
cat('There are', length(numericVars), 'numeric variables, and', length(factorVars), 'categoric variables')

all_numVar <- full_data[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
#select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)

# Random forrest to determine variable importance

quick_RF <- randomForest(x=full_data[1:1459,-79], y=full_data$SalePrice[1:1459], ntree=100,importance=TRUE)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]

ggplot(imp_DF[1:20,], aes(x=reorder(Variables, MSE), 
                          y=MSE, fill=MSE)) + 
  geom_bar(stat = 'identity') + 
  labs(x = "", y= '% increase MSE if variable is randomly permuted') + 
  coord_flip() + 
  theme(legend.position="none") +
  theme_minimal() +
  ggtitle("Most Important Variables") +
  theme(legend.position = "none")

#############################
# Feature engineering
#############################

# Total baths

full_data$TotBathrooms <- full_data$FullBath + 
  (full_data$HalfBath*0.5) + 
  full_data$BsmtFullBath + 
  (full_data$BsmtHalfBath*0.5)

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(TotBathrooms), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Number of Bathrooms and Selling Price") +
  theme(plot.title = element_text(hjust = 0.5))

# House age and dummy for remodeled

full_data$Remod <- ifelse(full_data$YearBuilt==full_data$YearRemodAdd, 0, 1) #0=No Remodeling, 1=Remodeling
full_data$Age <- as.numeric(full_data$YrSold)-full_data$YearRemodAdd

ggplot(full_data[!is.na(full_data$SalePrice),], aes(x=Age, y=SalePrice)) +
  geom_point(aes(alpha = .5)) +
  geom_smooth() +
  ggtitle("Age of House and Selling Price") +
  ylab("") +
  xlab("") +
  theme_minimal() +
  scale_y_continuous(labels = comma)+
  theme(plot.title = element_text(hjust = .5))

# Clearly a downward trend here. The older a house, the less it sells for, on average. 

cor(full_data$SalePrice[!is.na(full_data$SalePrice)], full_data$Age[!is.na(full_data$SalePrice)])

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(Remod), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("Remodeled and Selling Price") +
  theme(plot.title = element_text(hjust = 0.5))

# Median selling price is lower for houses that have been remodeled

# Adding dummy for new or not

full_data$IsNew <- ifelse(full_data$YrSold==full_data$YearBuilt, 1, 0)
table(full_data$IsNew)

ggplot(data=full_data[!is.na(full_data$SalePrice),], aes(x=factor(IsNew), y=SalePrice))+
  geom_boxplot(color = "#6D9EC1") + labs(x='', 
                                         y = '') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) + 
  theme_minimal() + 
  ggtitle("New vs. Used Houses") +
  theme(plot.title = element_text(hjust = 0.5))

full_data$YrSold <- as.factor(full_data$YrSold) # Can now change year to a factor variable

nb1 <- ggplot(full_data[!is.na(full_data$SalePrice),], aes(x=reorder(Neighborhood, SalePrice, FUN=median), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median") + labs(x='Neighborhood', y='Median SalePrice') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") 

nb2 <- ggplot(full_data[!is.na(full_data$SalePrice),], aes(x=reorder(Neighborhood, SalePrice, FUN=mean), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "mean") + labs(x='Neighborhood', y="Mean SalePrice") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red")

grid.arrange(nb1, nb2)

# There are 3 very expensive neighborhoods that stand out
# 3 very cheap neighborhoods
# and everything in between
# Binning neighborhoods

full_data$NeighRich[full_data$Neighborhood %in% c('StoneBr', 'NridgHt', 'NoRidge')] <- 2
full_data$NeighRich[!full_data$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale', 'StoneBr', 'NridgHt', 'NoRidge')] <- 1
full_data$NeighRich[full_data$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale')] <- 0

table(full_data$NeighRich)

# Total square feet

full_data$TotalSqFeet <- full_data$GrLivArea + full_data$TotalBsmtSF

ggplot(full_data[!is.na(full_data$SalePrice),], aes(x=TotalSqFeet, y=SalePrice)) +
  geom_point() +
  geom_smooth(method = "lm") +
  ggtitle("Total Square Footage and Sale Price") +
  ylab("") +
  xlab("") +
  theme_minimal() +
  scale_y_continuous(labels = comma)+
  theme(plot.title = element_text(hjust = .5))

cor(full_data$SalePrice, full_data$TotalSqFeet, use= "pairwise.complete.obs")

# Remove outliers

full_data[which(full_data$TotalSqFeet > 7500), c('TotalSqFeet', 'SalePrice')]

# observation 524, 1299 (training set) and 2550 (testing set) appear to be outliers
# check correlation without the training set observations

cor(full_data$SalePrice[-c(524, 1299)], full_data$TotalSqFeet[-c(524, 1299)], use= "pairwise.complete.obs")
full_data <- full_data[-c(524, 1299), ]

glimpse(full_data)

# Correlation is 5% higher with these two omitted

# Consolidating Porch variables

# WoodDeckSF: Wood deck area in square feet
# OpenPorchSF: Open porch area in square feet
# EnclosedPorch: Enclosed porch area in square feet
# 3SsnPorch: Three season porch area in square feet
# ScreenPorch: Screen porch area in square feet

full_data$TotalPorchSF <- full_data$OpenPorchSF + full_data$EnclosedPorch + full_data$X3SsnPorch + full_data$ScreenPorch

cor(full_data$SalePrice, full_data$TotalPorchSF, use= "pairwise.complete.obs")

# .195 correlation. Not very strong

ggplot(full_data[!is.na(full_data$SalePrice),], aes(x=TotalPorchSF, y=SalePrice)) +
  geom_point() +
  geom_smooth(method = "lm") +
  ggtitle("Total Porch Square Footage and Sale Price") +
  ylab("") +
  xlab("") +
  theme_minimal() +
  scale_y_continuous(labels = comma)+
  theme(plot.title = element_text(hjust = .5))

############################
# Prep the data for modeling
############################

# Drop highly correlated variables

dropVars <- c('YearRemodAdd', 'GarageYrBlt', 'GarageArea', 
              'GarageCond', 'TotalBsmtSF', 'TotalRmsAbvGrd', 
              'BsmtFinSF1')

full_data <- full_data[,!(names(full_data) %in% dropVars)]


# Scale and center data, create dummies for categorical vars

numericVarNames <- numericVarNames[!(numericVarNames %in% c('MSSubClass', 'MoSold', 
                                                            'YrSold', 'SalePrice', 
                                                            'OverallQual', 'OverallCond'))] #numericVarNames was created before having done anything
numericVarNames <- append(numericVarNames, c('Age', 'TotalPorchSF', 'TotBathrooms', 'TotalSqFeet'))

DFnumeric <- full_data[, names(full_data) %in% numericVarNames]

DFfactors <- full_data[, !(names(full_data) %in% numericVarNames)]
DFfactors <- DFfactors[, names(DFfactors) != 'SalePrice']

cat('There are', length(DFnumeric), 'numeric variables, and', length(DFfactors), 'factor variables')

# Taking log of all numeric predictors with an absolute skew over .8

for(i in 1:ncol(DFnumeric)){
  if (abs(skew(DFnumeric[,i]))>0.8){
    DFnumeric[,i] <- log(DFnumeric[,i] +1)
  }
}

PreNum <- preProcess(DFnumeric, method=c("center", "scale"))
print(PreNum)

DFnorm <- predict(PreNum, DFnumeric)
dim(DFnorm)

# One-hot encoding categorical variables (dummy dataframe)

DFdummies <- as.data.frame(model.matrix(~.-1, DFfactors))
dim(DFdummies)

# are there variables with zero observations in the test set?

ZerocolTest <- which(colSums(DFdummies[(nrow(full_data[!is.na(full_data$SalePrice),])+1):nrow(full_data),])==0)
colnames(DFdummies[ZerocolTest])

DFdummies <- DFdummies[,-ZerocolTest] #removing predictors

# check if some values are absent in the train set

ZerocolTrain <- which(colSums(DFdummies[1:nrow(full_data[!is.na(full_data$SalePrice),]),])==0)
colnames(DFdummies[ZerocolTrain])

DFdummies <- DFdummies[,-ZerocolTrain] #removing predictor

# taking out vars with less than 10 true values

fewOnes <- which(colSums(DFdummies[1:nrow(full_data[!is.na(full_data$SalePrice),]),])<10)
colnames(DFdummies[fewOnes])

DFdummies <- DFdummies[,-fewOnes] #removing predictors
dim(DFdummies)

combined <- cbind(DFnorm, DFdummies)

# Dealing with scewness of response variable

skew(full_data$SalePrice)

qqnorm(full_data$SalePrice)
qqline(full_data$SalePrice)

# Not normally distributed. Scewed. Taking log

full_data$SalePrice <- log(full_data$SalePrice) #default is the natural logarithm, "+1" is not necessary as there are no 0's
skew(full_data$SalePrice)

qqnorm(full_data$SalePrice)
qqline(full_data$SalePrice) # Better

train1 <- combined[!is.na(full_data$SalePrice),]
test1 <- combined[is.na(full_data$SalePrice),]

# Adding interaction terms

#####################################
# MODELING 
####################################

# LASSO 

my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))

lasso_mod <- train(x=train1, y=full_data$SalePrice[!is.na(full_data$SalePrice)], method='glmnet', trControl= my_control, tuneGrid=lassoGrid) 
lasso_mod$bestTune

min(lasso_mod$results$RMSE)

lassoVarImp <- varImp(lasso_mod,scale=F)
lassoImportance <- lassoVarImp$importance

varsSelected <- length(which(lassoImportance$Overall!=0))
varsNotSelected <- length(which(lassoImportance$Overall==0))

cat('Lasso uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')

LassoPred <- predict(lasso_mod, test1)
predictions_lasso <- exp(LassoPred) #need to reverse the log to the real values
head(predictions_lasso)

# XGBoost

xgb_grid = expand.grid(
  nrounds = 1000,
  eta = c(0.1, 0.05, 0.01),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree=1,
  min_child_weight=c(1, 2, 3, 4 ,5),
  subsample=1
)

# The code below takes QUITE A WHILE TO RUN
# Let caret find the best hyperparameter values. 5-fold cross-validation

# xgb_caret <- train(x=train1, y=full_data$SalePrice[!is.na(full_data$SalePrice)], method='xgbTree', trControl= my_control, tuneGrid=xgb_grid) 
# xgb_caret$bestTune

# max depth 3
# eta .05
# min child weight 4

label_train <- full_data$SalePrice[!is.na(full_data$SalePrice)]

# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = as.matrix(train1), label= label_train)
dtest <- xgb.DMatrix(data = as.matrix(test1))

default_param<-list(
  objective = "reg:linear",
  booster = "gbtree",
  eta=0.05, #default = 0.3
  gamma=0,
  max_depth=3, #default=6
  min_child_weight=4, #default=1
  subsample=1,
  colsample_bytree=1
)

# Cross validation to determine the best number of rounds

xgbcv <- xgb.cv( params = default_param, data = dtrain, 
                 nrounds = 500, nfold = 5, showsd = T, 
                 stratified = T, print_every_n = 40, early_stopping_rounds = 10, 
                 maximize = F)

xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 382)

XGBpred <- predict(xgb_mod, dtest)
predictions_XGB <- exp(XGBpred) #need to reverse the log to the real values
head(predictions_XGB)

library(Ckmeans.1d.dp)

mat <- xgb.importance (feature_names = colnames(train1),model = xgb_mod)
xgb.ggplot.importance(importance_matrix = mat[1:30], rel_to_first = TRUE)

# Averaging the results

sub_avg <- data.frame(Id = test_labels, SalePrice = (predictions_XGB+predictions_lasso)/2)
head(sub_avg)

write.csv(sub_avg, "kaggle_house_price_predictions.csv", row.names = FALSE)
