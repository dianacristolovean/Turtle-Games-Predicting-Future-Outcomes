## LSE Data Analytics Online Career Accelerator 
# DA301:  Advanced Analytics for Organisational Impact

###############################################################################

# Assignment 5 scenario
## Turtle Games' sales department has historically preferred to use R when performing 
## sales analyses due to existing workflow systems. As you’re able to perform data analysis 
## in R, you will perform exploratory data analysis and present your findings by utilizing 
## basic statistics and plots. You'll explore and prepare the data set to analyse sales per 
## product. The sales department is hoping to use the findings of this exploratory analysis 
## to inform changes and improvements in the team. (Note that you will use basic summary 
## statistics in Module 5 and will continue to go into more detail with descriptive 
## statistics in Module 6.)

################################################################################

## Assignment 5 objective
## Load and wrangle the data. Use summary statistics and groupings if required to sense-check
## and gain insights into the data. Make sure to use different visualizations such as scatterplots, 
## histograms, and boxplots to learn more about the data set. Explore the data and comment on the 
## insights gained from your exploratory data analysis. For example, outliers, missing values, 
## and distribution of data. Also make sure to comment on initial patterns and distributions or 
## behavior that may be of interest to the business.

################################################################################

# Module 5 assignment: Load, clean and wrangle data using R

## It is strongly advised that you use the cleaned version of the data set that you created and 
##  saved in the Python section of the course. Should you choose to redo the data cleaning in R, 
##  make sure to apply the same transformations as you will have to potentially compare the results.
##  (Note: Manual steps included dropping and renaming the columns as per the instructions in module 1.
##  Drop ‘language’ and ‘platform’ and rename ‘remuneration’ and ‘spending_score’) 

## 1. Open your RStudio and start setting up your R environment. 
## 2. Open a new R script and import the turtle_review.csv data file, which you can download from 
##      Assignment: Predicting future outcomes. (Note: You can use the clean version of the data 
##      you saved as csv in module 1, or, can manually drop and rename the columns as per the instructions 
##      in module 1. Drop ‘language’ and ‘platform’ and rename ‘remuneration’ and ‘spending_score’) 
## 3. Import all the required libraries for the analysis and view the data. 
## 4. Load and explore the data.
##    - View the head the data.
##    - Create a summary of the new data frame.
## 5. Perform exploratory data analysis by creating tables and visualisations to better understand 
##      groupings and different perspectives into customer behaviour and specifically how loyalty 
##      points are accumulated. Example questions could include:
##    - Can you comment on distributions, patterns or outliers based on the visual exploration of the data?
##    - Are there any insights based on the basic observations that may require further investigation?
##    - Are there any groupings that may be useful in gaining deeper insights into customer behaviour?
##    - Are there any specific patterns that you want to investigate
## 6. Create
##    - Create scatterplots, histograms, and boxplots to visually explore the loyalty_points data.
##    - Select appropriate visualisations to communicate relevant findings and insights to the business.
## 7. Note your observations and recommendations to the technical and business users.

###############################################################################

# Your code here.

## Install and import the necessary packages and libraries.

# install.packages("stringr")
# install.packages('tidyverse')
# install.packages("readxl")
# install.packages("readr")
# install.packages("openxlsx")
# install.packages("ggplot2")
# install.packages(c("skimr", "DataExplorer", "scales", "plotly"))
# install.packages(c("moments", "psych"))
# install.packages("corrplot")
# install.packages("dplyr")

library(tidyverse)
library(ggplot2)
library(dplyr)
library (skimr)
library(DataExplorer)
library(scales)
library(plotly)
library(corrplot)
library(moments)
library(psych)
# library(openxlsx)

###############################################################################
# Define functions

# Create a scatterplot
plot_scatter <- function(data, x_var, y_var, x_label, y_label, title) {

  ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "lm", se = FALSE, color = 'Red') +
    labs(title = title, x = x_label, y = y_label) +
    theme_minimal()
}

# Create histogram 
plot_density <- function(data, x_var, title, x_label, x_limits = c(0, 120), x_breaks = seq(0, 120, by = 20)) {
  ggplot(data, aes_string(x = x_var)) +
    geom_density(fill = "blue", color = "black", alpha = 0.5) +
    labs(title = title, x = x_label, y = "Density") +
    scale_x_continuous(limits = x_limits, breaks = x_breaks)
}

###############################################################################
# Import the previously cleaned data file reviews.csv
data <- read.csv(file.choose(),header=TRUE)

#Sense Check Data 
head(data)
str(data)

# Check missing values
missing_values <- sum(is.na(data))
print(missing_values)

# Check duplicated values 
num_duplicates <- sum(duplicated(data))
print(num_duplicates)

# Look at summary
summary(data)

###############################################################################

# Determine mean value variable for plotting 
mean_loyalty_points <- mean(data$loyalty_points)
median_loyalty_points <- median(data$loyalty_points)

# Visualise Loyalty points Distribution with mean line

ggplot(data, aes(x = loyalty_points)) +
  geom_histogram(binwidth = 200, fill = "blue", color = "red", alpha = 0.6) +
  geom_vline(aes(xintercept = median_loyalty_points, color = "Median"), linetype = "dashed", size = 1) +
  labs(title = "Distribution of Loyalty Points", x = "Loyalty Points", y = "Frequency") +
  scale_x_continuous(breaks = seq(0, max(data$loyalty_points), by = 1000)) +
  scale_color_manual(name = "Statistics", values = c("Median" = "red")) +
  theme_minimal()

# Skewness Test 
skewness(data$loyalty_points)
# The distribution of Loyalty Points is right-skewed (positively skewed).
# The median (red dashed line) is to the left of the tail, 
# which is also a visual indicator of right-skewness.
# This means:
#   
# - Most customers have relatively low loyalty points.
# - There’s a smaller number of customers with very high loyalty points, 
# pulling the tail to the right.
# - Median < Mean (another classic trait of right-skewed distributions).

# Shapiro-Wilk test for normality
shapiro.test(data$loyalty_points)

# Kurtosis Test 
kurtosis(data$loyalty_points)

# From our exploratory analysis we can conclude that loyalty points are not 
# normally distributed (w= 0.84), are positively skewed (skewness= 1.46) and
# exhibit heavy tail and sharper peak (Kurtosis = 4.70), suggesting the presence 
# of outlier and a concentration of values around the mean.

# Boxplot to identify potential outliers
ggplot(data, aes(y = loyalty_points)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Loyalty Points", y = "Loyalty Points") +
  scale_y_continuous(limits = c(0, 7000), breaks = seq(0, 7000, by = 200))
# Loyalty points present several outliers on the right whisker, 
# suggesting that some customers have significantly higher loyalty points 
# compared to others 

# Boxplot to identify potential outliers
ggplot(data, aes(x = education, y = loyalty_points, fill = gender)) +
  geom_boxplot() +
  labs(title = "Loyalty Points vs Education across Genders", x = "Education", y = "Loyalty Points", fill = "Gender") +
  scale_y_continuous(limits = c(0, 7000), breaks = seq(0, 7000, by = 200))
# - Outliers are more frequent in higher education levels, 
#   but the "Basic" group has the largest IQR and the highest extreme values, 
#   especially among females.
# - This suggests that education level does not strictly limit high loyalty behavior, 
#   and outliers can occur regardless of education—but are most visually dominant 
#   in the "Basic" group.
# - Outliers are mostly high values, consistent with the earlier observed 
#   right-skewed distribution.

###############################################################################

# Visual exploration of spend_score distribution
plot_density(data, "spending_score", "Distribution of Spending Score", "Spending Score", x_limits = c(0, 100), x_breaks = seq(0, 100, by = 10))

# Boxplot to identify potential outliers
ggplot(data, aes(y = spending_score)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Spending Score", y = "Spending Score") +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 10))

# Boxplot to identify potential outliers
ggplot(data, aes(x = education, y = spending_score, fill = gender)) +
  geom_boxplot() +
  labs(title = "Boxplot of Spending Score vs Education across Genders", x="Education", y = "Spending Score", fill = "Gender") +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 10))
# - There are not outliers in spending score

###############################################################################

# Histogram of income
plot_density(data, "income", "Distribution of Income", "Income (k£)", x_limits = c(0, 120), x_breaks = seq(0, 120, by = 20))

# Boxplot of income
ggplot(data, aes(y = income)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Income", y = "Income") +
  scale_y_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 20))

# Boxplot to identify potential outliers
ggplot(data, aes(x = education, y = income, fill=gender)) +
  geom_boxplot() +
  labs(title = "Boxplot of Remuneration vs Education across Genders", x="Education", y = "Income", fill = "Gender") +
  scale_y_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 20))
# - Very few outliers observed

###############################################################################

# Histogram of age with bin size of 20
ggplot(data, aes(x = age)) +
  geom_histogram(bins = 20, fill = "blue", color = "black", na.rm = TRUE) +
  labs(title = "Distribution of Age", x = "Age", y = "Frequency") +
  scale_x_continuous(limits = c(0, 80), breaks = seq(0, 80, by = 10))

# Generate box plot of age
ggplot(data, aes(y = age)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Age", y = "Age") +
  scale_y_continuous(limits = c(0, 80), breaks = seq(0, 80, by = 10))

# Boxplot to identify potential outliers
ggplot(data, aes(x = education, y = age, fill = gender)) +
  geom_boxplot() +
  labs(title = "Boxplot of Age vs Educations across Genders", x = "Education", y = "Age", fill = "Gender") +
  scale_y_continuous(limits = c(0, 80), breaks = seq(0, 80, by = 10))
# - Very few outliers observed

###############################################################################

# Visualise how loyalty points relate to Income, Spending Score and Age.

# Scatterplot of Loyalty Points  vs Spending Score
plot_scatter(data, "spending_score", "loyalty_points", "Spending Score", "Loyalty Points", "Loyalty Points vs Spending Score")

# Scatterplot of Loyalty Points vs Income

plot_scatter(data, "income", "loyalty_points", "Income (k£)", "Loyalty Points", "Loyalty Points vs Income")

# Scatterplot of Loyalty Points  vs Age
plot_scatter(data, "age", "loyalty_points", "Age", "Loyalty Points", "Loyalty Points vs Age")

###############################################################################

# Create a table of education levels and customer counts
education_counts <- table(data$education)

# Convert the table to a data frame for plotting
education_counts_df <- as.data.frame(education_counts)
head(education_counts_df)

# Rename the columns for clarity
colnames(education_counts_df) <- c("education", "count")
head(education_counts_df)

# Create the barplot
ggplot(education_counts_df, aes(x = education, y = count)) +
  geom_bar(stat = "identity", fill = "blue", color = "black") +
  labs(title = "Number of customers by education Level",
       x = "Education Level",
       y = "Number of Customers") +
  theme_minimal()

###############################################################################

# Because outliers observed in loyalty points at 3200 and above, 
# I have decided to investigate a little how many customers are above this threshold

# Create a new column indicating whether loyalty points are above or below the threshold of 3200
data$loyalty_group <- ifelse(data$loyalty_points >= 3200, "3200 and Above", "Below 3200")

# Calculate the counts for each group
loyalty_counts <- table(data$loyalty_group)

# Convert the table to a data frame for plotting
loyalty_counts_df <- as.data.frame(loyalty_counts)

# Rename columns for clarity
colnames(loyalty_counts_df) <- c("Loyalty_Group", "Count")

# Calculate proportions
loyalty_counts_df$Proportion <- loyalty_counts_df$Count / sum(loyalty_counts_df$Count)

# Format proportions as percentages
loyalty_counts_df$Proportion_Percent <- paste0(round(loyalty_counts_df$Proportion * 100, 1), "%")

# Create the bar plot
ggplot(loyalty_counts_df, aes(x = Loyalty_Group, y = Count, fill = Loyalty_Group)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(Count, " (", Proportion_Percent, ")")), vjust = -0.5) + # Add count labels above each bar
  labs(title = "Number of Customers by Loyalty Points",
       x = "Loyalty Point Group",
       y = "Number of Customers") +
  theme_minimal()

###############################################################################

# Drop unnecessary columns for correlation analysis 
numeric_data <- data %>%
  select_if(is.numeric)

# Calculate the correlation matrix
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# View Matrix 
print(correlation_matrix)

# Create the correlation plot
corrplot(correlation_matrix, method = "circle")

# Pair plots to visualize relationships
pairs(data[, c('age', 'income', 'spending_score', 'loyalty_points', 'product')])

# Key Observations and Insights

# Loyalty Points are right-skewed, with most customers having low points 
# and a few with very high points.
# 
# Outliers above 3,200 points suggest a highly loyal segment worth further analysis.
# 
# Spending Score shows a strong positive correlation with loyalty points
# —higher spending leads to higher loyalty.
# 
# Income also correlates positively, but less strongly than spending score.
# 
# Age has a weak correlation; older customers have slightly more points.
# 
# Education level shows minor variation, with higher education 
# slightly linked to more points.
# 
# Gender does not significantly impact loyalty point accumulation.
# 
# These relationships suggest some degree of linearity, although the data becomes more dispersed
# for customers with a spending score above 60 or a higher yearly income.
#
# Segmenting by loyalty level (above/below 3,200) shows higher loyalty 
# is linked to higher income and spending, and a slightly younger age.
# 
# The loyalty program works well for high-spending customers but could be improved 
# for lower-tier segments.


###############################################################################
###############################################################################

# Assignment 6 scenario

## In Module 5, you were requested to redo components of the analysis using Turtle Games’s preferred 
## language, R, in order to make it easier for them to implement your analysis internally. As a final 
## task the team asked you to perform a statistical analysis and create a multiple linear regression 
## model using R to predict loyalty points using the available features in a multiple linear model. 
## They did not prescribe which features to use and you can therefore use insights from previous modules 
## as well as your statistical analysis to make recommendations regarding suitability of this model type,
## the specifics of the model you created and alternative solutions. As a final task they also requested 
## your observations and recommendations regarding the current loyalty programme and how this could be 
## improved. 

################################################################################

## Assignment 6 objective
## You need to investigate customer behaviour and the effectiveness of the current loyalty program based 
## on the work completed in modules 1-5 as well as the statistical analysis and modelling efforts of module 6.
##  - Can we predict loyalty points given the existing features using a relatively simple MLR model?
##  - Do you have confidence in the model results (Goodness of fit evaluation)
##  - Where should the business focus their marketing efforts?
##  - How could the loyalty program be improved?
##  - How could the analysis be improved?

################################################################################

## Assignment 6 assignment: Making recommendations to the business.

## 1. Continue with your R script in RStudio from Assignment Activity 5: Cleaning, manipulating, and 
##     visualising the data.
## 2. Load and explore the data, and continue to use the data frame you prepared in Module 5.
## 3. Perform a statistical analysis and comment on the descriptive statistics in the context of the 
##     review of how customers accumulate loyalty points.
##  - Comment on distributions and patterns observed in the data.
##  - Determine and justify the features to be used in a multiple linear regression model and potential
##.    concerns and corrective actions.
## 4. Create a Multiple linear regression model using your selected (numeric) features.
##  - Evaluate the goodness of fit and interpret the model summary statistics.
##  - Create a visual demonstration of the model
##  - Comment on the usefulness of the model, potential improvements and alternate suggestions that could 
##     be considered.
##  - Demonstrate how the model could be used to predict given specific scenarios. (You can create your own 
##     scenarios).
## 5. Perform exploratory data analysis by using statistical analysis methods and comment on the descriptive 
##     statistics in the context of the review of how customers accumulate loyalty points.
## 6. Document your observations, interpretations, and suggestions based on each of the models created in 
##     your notebook. (This will serve as input to your summary and final submission at the end of the course.)

################################################################################

# Your code here.

# Create the multiple linear regression model
model <- lm(loyalty_points ~ income + spending_score, data = data)

summary(model)

# Model Fit & Performance
# Residual Standard Error: 534.1
# - On average, the model’s predictions are off by ~534 units.
# 
# Multiple R-squared: 0.8269
# - 82.69% of the variance in the dependent variable is explained by the model.
# 
# Adjusted R-squared: 0.8267
# → Adjusts for the number of predictors, indicating both predictors are meaningful.
# 
# F-statistic: 4770 on 2 and 1997 DF
# - Extremely strong model overall.
# 
# p-value: < 2.2e-16
# - The model is highly statistically significant.
#
# Median Residual: 40.34 → Errors are centered near 0, which is good.
# 
# Range of Residuals: From -1646.02 to +1999.95 → Some large prediction errors (possible outliers).
# 
# Interquartile Range (IQR): 50% of errors fall between -363.66 and +280.59 
#   → Shows typical prediction spread.
#
# All predictors are highly significant (p < 2e-16) with very strong positive effects.
# 
# t-values for both predictors are large (65.77 and 71.84), confirming their strong impact.

# Get the model residuals
model_residuals = model$residuals

# Plot the result
hist(model_residuals)


# Plot residuals
plot(model$residuals)

#add a horizontal line at 0 
abline(0,0)


# Visualize Model Performance

# Plot actual vs. predicted values 
ggplot(data, aes(x = loyalty_points, y = predict(model, data))) +
  geom_point() +
  stat_smooth(method = "loess") + 
  labs(x = 'Observed Loyalty Points', y = 'Predicted Loyalty Points') +
  ggtitle('Observed vs. Predicted Loyalty Points')+
  theme_minimal()

# The residuals are not normally distributed. Although the model shows a strong correlation 
# and a high R-squared value, the fit is poor due to a partially non-linear relationship 
# between the predictors and the target variable.

###############################################################################

# To address this, we will apply a transformation to the target variable.

data <- mutate(data, log_loyalty_points = log(loyalty_points))

# Check distribution of transformed loyalty points
hist(data$log_loyalty_points)

# Check normality 
# Shapiro-Wilk test for normality
shapiro.test(data$log_loyalty_points)

# Skewness Test 
skewness(data$log_loyalty_points)

# Kurtosis Test 
kurtosis(data$log_loyalty_points)

# Q-Q Plot
qqnorm(data$log_loyalty_points)
qqline(data$log_loyalty_points, col = "blue")

# Even after a log transformation, log_loyalty_points is still non-normal 
# (confirmed by Shapiro-Wilk and Q-Q plot)
#
# Since the p-value is extremely low, you reject the null hypothesis 
# - the data is not normally distributed, even after log transformation.
# Left-skewed (-1.19 is moderately to heavily skewed.)
# 
# Has heavy tails (high kurtosis) 
# - our data has more extreme values (outliers) than a normal distribution would.

# MLR on transformed variable
model_2 <- lm(log_loyalty_points ~ income + spending_score, data = data)
summary(model_2)

# Predict on the log-transformed model
predicted_loyalty_log <- predict(model_2, data)

# Transform predictions back to the original scale
predicted_original_scale <- exp(predicted_loyalty_log)

# Plot Residuals  
ggplot(data, aes(x = loyalty_points, y = predicted_original_scale)) +
  geom_point() +
  stat_smooth(method = "loess") + 
  labs(x = 'Observed Loyalty Points', y = 'Predicted Loyalty Points') +
  ggtitle('Observed vs. Predicted Loyalty Points')

plot(model_2, which = 2)  # Residual Q-Q plot
# The pattern suggests left-skewed and heavy-tailed residuals, 
# consistent with the earlier log-transformed variable being left-skewed.

shapiro.test(residuals(model_2))
# Extremely low p-value means we reject the null (residuals are not normally distributed).
# This is a key issue because normality of residuals is an important assumption in linear regression 
# (especially for inference, confidence intervals, and p-values).

plot(model_2, which = 1)  # Residuals vs Fitted
plot(model_2, which = 3)  # Scale-Location
# This suggests mild to moderate heteroscedasticity 
# — the variance of residuals is not perfectly constant, 
# particularly at the extremes of fitted values.
# The residuals show signs of heteroscedasticity, especially at low/high predicted values.
# This weakens the assumptions of the linear model.

# The model has a good R², but the residuals are not normally distributed, 
# even after log-transforming the target variable.
# 
# This may affect statistical inference (like t-tests and p-values).
# 
# The model is not fully capturing the true relationship, 
# or there's a non-linear structure or outliers still present.

###############################################################################

# Check Predictions with new data

# Example with 3 new customer: 
# Customer a: Income =25, Spending Score = 20
# Customer b: Income =45, Spending Score = 70
# Customer c: Income =60, Spending Score = 60

new_data <- data.frame(income = c(25,45,60), spending_score = c(20,70,60))

# Predict Loyalty Points for new customers
predictions <- predict(model_2, new_data)
print(predictions)

###############################################################################
###############################################################################




