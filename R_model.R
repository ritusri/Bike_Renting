# importing all required library
required_library <- c('ggplot2', 'corrgram', 'corrplot', 'randomForest',
                      'caret', 'class', 'e1071', 'rpart', 'mlr','grid',
                      'DMwR','usdm','dplyr','caTools','LiblineaR')

# checking for each library whether installed or not
# if not install then installing it first and then attaching to file
for (lib in required_library){
  if(!require(lib, character.only = TRUE))
  {
    install.packages(lib)
    require(lib, character.only = TRUE)
  }
}

# removing extra variable
rm(required_library,lib)

# set working directory to the file location, uncomment below line and put full path
# setwd("full path to folder in which file is present")
# reading csv file
bike_data = pd.read_csv("C:/Users/anupr/Desktop/projects/bike renting/ day.csv")

############################################
#                                          #
#     2.1 Exploratory Data Analysis        #
#                                          #
############################################

###################################
#  2.1.1 understanding the data   #
###################################

# cheking datatypes of all columns
str(bike_data)

numeric_columns <- c('temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt')
cat_columns <- c('season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit')

### checking numerical variables ###
# Checking numerical statistics of numerical columns (Five point summary + mean of all column)
summary(bike_data[,numeric_columns])

### Checking categorical variable ###
# unique values in each category
lapply(bike_data[,cat_columns], function(feat) length(unique(feat)))

# counting of each unique values in categorical columns
lapply(bike_data[,cat_columns], function(feature) table(feature))

###################################
#  2.1.2 Missing value analysis   #
###################################

# checking missing value for each column and storing counting in dataframe with column name
missing_val <- data.frame(lapply(bike_data, function(feat) sum(is.na(feat))))

###################################
#  2.1.3 outlier analysis         #
###################################

# box_plot function to plot boxplot of numerical columns
box_plot <- function(column, dataset){
  dataset$x = 1
  ggplot(aes_string(x= 'x', y = column), data = dataset)+
    stat_boxplot(geom = 'errorbar', width = 0.5)+
    geom_boxplot(outlier.size = 2, outlier.shape = 18)+
    labs(y = "", x = column)+
    ggtitle(paste(" BP :",column))
}

# hist_plot function to plot histogram of numerical variable
hist_plot <- function(column, dataset){
  ggplot(aes_string(column), data = dataset)+
    geom_histogram(aes(y=..density..), fill = 'skyblue2')+
    geom_density()+
    labs(x = gsub('\\.', ' ', column))+
    ggtitle(paste(" Histogram :",gsub('\\.', ' ', column)))
}

# calling box_plot function and storing all plots in a list
all_box_plots <- lapply(c('temp', 'atemp', 'hum', 'windspeed'),box_plot, dataset = bike_data)

# calling hist_plot function and storing all plots in a list
all_hist_plots <- lapply(c('temp', 'atemp', 'hum', 'windspeed'),hist_plot, dataset = bike_data)

# printing all plots in one go
gridExtra::grid.arrange(all_box_plots[[1]],all_box_plots[[2]],all_box_plots[[3]],all_box_plots[[4]],
                        all_hist_plots[[1]],all_hist_plots[[2]],all_hist_plots[[3]],all_hist_plots[[4]],ncol=4,nrow=2)


###################################
#  2.1.4 Feature Engineering      #
###################################

# method which will plot barplot of a columns with respect to other column
plot_bar <- function(cat, y, fun){
  gp = aggregate(x = bike_data[, y], by=list(cat=bike_data[, cat]), FUN=fun)
  ggplot(gp, aes_string(x = 'cat', y = 'x'))+
    geom_bar(stat = 'identity')+
    labs(y = y, x = cat)+
    ggtitle(paste("Bar plot for",y,"wrt to",cat))
}

# plotting cnt with respect to month
plot_bar('mnth', 'cnt', 'sum')

# plotting cnt with respect to yr
plot_bar('yr', 'cnt', 'sum')

# plotting cnt with respect to yr
plot_bar('weekday', 'cnt', 'sum')

# making bins of mnth and weekday
# changing values of month 5th to 10th as 1 and others 0
bike_data = transform(bike_data, mnth = case_when(
  mnth <= 4 ~ 0, 
  mnth >= 11 ~ 0,
  TRUE   ~ 1 
))
colnames(bike_data)[5] <- 'month_feat'

# changing values of weekday for day 0 and 1 the value will be 0
#and 1 for rest
bike_data = transform(bike_data, weekday = case_when(
  weekday < 2 ~ 0, 
  TRUE   ~ 1 
))
colnames(bike_data)[7] <- 'week_feat'

###################################
#  2.1.5 Feature Selection        #
###################################

# correlation plot for numerical feature
corrgram(bike_data[,numeric_columns], order = FALSE,
         upper.panel = panel.pie, text.panel = panel.txt,
         main = "Correlation Plot for bike data set")

# heatmap plot for numerical features
corrplot(cor(bike_data[,numeric_columns]), method = 'color', type = 'lower')

cat_columns <- c('season', 'yr', 'month_feat', 'holiday', 'week_feat', 'workingday', 'weathersit')

# making every combination from cat_columns
combined_cat <- combn(cat_columns, 2, simplify = F)

# doing chi-square test for every combination
for(i in combined_cat){
  print(i)
  print(chisq.test(table(bike_data[,i[1]], bike_data[,i[2]])))
}

# finding important features
important_feat <- randomForest(cnt ~ ., data = bike_data[,-c(1,2,14,15)],
                               ntree = 200, keep.forest = FALSE, importance = TRUE)
importance_feat_df <- data.frame(importance(important_feat, type = 1))

# checking vif of numerical column withhout dropping multicollinear column
vif(bike_data[,c(10,11,12,13)])

# Checking VIF values of numeric columns after dropping multicollinear column i.e. atemp
vif(bike_data[,c(10,12,13)])

# Making factor datatype to each category
bike_data[,cat_columns] <- lapply(bike_data[,cat_columns], as.factor)

# releasing memory of R, removing all variables except dataset
rm(list = setdiff(ls(),"bike_data"))

###################################
#  2.1.7 Data after EDA           #
###################################
# creating another dataset with dropping outliers i.e. bike_data_wo
bike_data_wo <- bike_data

# removing outliers from hum and windspeed columns
for (i in c('hum', 'windspeed')){
  out_value = bike_data_wo[,i] [bike_data_wo[,i] %in% boxplot.stats(bike_data_wo[,i])$out]
  bike_data_wo = bike_data_wo[which(!bike_data_wo[,i] %in% out_value),]
}

# checking dimension of both dataset
dim(bike_data)
dim(bike_data_wo)

# dropping unwanted columns
drop_col <- c('instant', 'dteday', 'holiday', 'atemp', 'casual', 'registered')
bike_data[,drop_col]<- NULL
bike_data_wo[,drop_col] <- NULL

############################################
#                                          #
#                                          #
#   2.2.2 Building models                  #
#                                          #
#                                          #
############################################
set.seed(1)
split = sample.split(bike_data$cnt, SplitRatio = 0.80)
train_set = subset(bike_data, split == TRUE)
test_set = subset(bike_data, split == FALSE)

split = sample.split(bike_data_wo$cnt, SplitRatio = 0.80)
train_set_wo = subset(bike_data_wo, split == TRUE)
test_set_wo = subset(bike_data_wo, split == FALSE)

# making a function which will train model on training data and would show 
# K-fold R2 score , R2 score for test dataset and train dataset
fit.predict.show.performance <- function(method, train_data, test_data){
  reg_fit <- caret::train(cnt~., data = train_data, method = method)
  
  y_pred <- predict(reg_fit, test_data[,-10])
  print("R2 on test dataset")
  print(caret::R2(y_pred, test_data[,10])) 
  
  y_pred <- predict(reg_fit, train_data[,-10])
  print("R2 on train dataset")
  print(caret::R2(y_pred, train_data[,10]))
  
  # creating 10 folds of data
  ten_folds = createFolds(train_data$cnt, k = 10)
  ten_cv = lapply(ten_folds, function(fold) {
    training_fold = train_data[-fold, ]
    test_fold = train_data[fold, ]
    reg_fit <- caret::train(cnt~., data = training_fold, method = method)
    
    y_pred <- predict(reg_fit, test_fold[,-10])
    return(as.numeric(caret::R2(y_pred, test_fold[,10]))) 
  })
    sum = 0
    for(i in ten_cv){
      sum = sum + as.numeric(i)
    }
    print("K-fold (K =10) explained variance")
    print(sum/10)
}
    

#########################
#   Linear Regression   #
#########################

# building model for dataset bike_data
fit.predict.show.performance('lm', train_set, test_set)

# building model for dataset bike_data_wo i.e. without  outliers
fit.predict.show.performance('lm', train_set_wo, test_set_wo)

#########################
#         KNN           #
#########################

# building model for dataset bike_data
fit.predict.show.performance('knn', train_set, test_set)

# building model for dataset bike_data_wo i.e. without  outliers
fit.predict.show.performance('knn', train_set_wo, test_set_wo)

#########################
#        SVM            #
#########################

# building model for dataset bike_data
fit.predict.show.performance('svmLinear3', train_set, test_set)

# building model for dataset bike_data_wo i.e. without  outliers
fit.predict.show.performance('svmLinear3', train_set_wo, test_set_wo)

#############################
# Decision Tree Regression  #
#############################

# building model for dataset bike_data
fit.predict.show.performance('rpart2', train_set, test_set)

# building model for dataset bike_data_wo i.e. without  outliers
fit.predict.show.performance('rpart2', train_set_wo, test_set_wo)

#########################
#  Random Forest        #
#########################

# building model for dataset bike_data
fit.predict.show.performance('rf', train_set, test_set)

# building model for dataset bike_data_wo i.e. without  outliers
fit.predict.show.performance('rf', train_set_wo, test_set_wo)

#########################
#     XGBRegressor      #
#########################

# building model for dataset bike_data
fit.predict.show.performance('xgbTree', train_set, test_set)

# building model for dataset bike_data_wo i.e. without  outliers
fit.predict.show.performance('xgbTree', train_set_wo, test_set_wo)


############################################
#                                          #
#                                          #
#        Hyperparameter tuning             #
#                                          #
#                                          #
############################################
###############################################
#                                             #
# tuning Random Forest for bike_data dataset  #
#                                             #
###############################################

control <- trainControl(method="repeatedcv", number=10, repeats=3)
reg_fit <- caret::train(cnt~., data = train_set, method = "rf",trControl = control)
reg_fit$bestTune
y_pred <- predict(reg_fit, test_set[,-10])
print(caret::R2(y_pred, test_set[,10]))

###############################################
#                                             #
#      tuning XGB for bike_data dataset       #
#                                             #
###############################################

control <- trainControl(method="repeatedcv", number=10, repeats=3)
reg_fit <- caret::train(cnt~., data = train_set, method = "xgbTree",trControl = control)
reg_fit$bestTune
y_pred <- predict(reg_fit, test_set[,-10])
print(caret::R2(y_pred, test_set[,10]))


