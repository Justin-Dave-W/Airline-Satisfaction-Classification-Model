#  Predicting Customer Satisfaction in Airline Industry


## General Overview
**Goal:**
- Create a classification model capable of predicting customer satisfaction in the form of a binary categorical variable (satisfied/not satisfied)
- Identify prominent factors from the model that affects customer satisfaction classification

**Data Source:**
- https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data
- Shape = 129,880 rows and 24 columns
- Variable Examples: Flight distance, customer rating from 0 to 5 on wifi and online booking services, age, departure delay in minutes

**Models Tested:**
- Naive Bayes
- Random Forest
- Decision Tree

**Model Optimization Methodologies:**
- Synthetic Minority Oversampling Technique (SMOTE)
- Hyperparameter Tuning using GridSearch
- Cost Complexity Pruning
- Wrapper Method Feature Selection
- Binning Contineous Variables


## Table of Contents 
- [Data](#data)
- [Exploratory Analysis](#exploratory-analysis)
- [Model Benchmarking and Optimization](#model-benchmarking-and-optimization)
- [Model Interpretation](#model-interpretation)
- [Final Remarks](#final-remarks)

## Data

The following table describes the attributes present in the dataset before any data pre pre processing took place:

![image](https://github.com/user-attachments/assets/6577ddc0-0bab-467c-8755-716d753598c8)


## Exploratory Analysis

### Distribution of Numerical Variables
![image](https://github.com/user-attachments/assets/86a391d8-74c5-4bef-b2b7-95e1cafff50e)

![image](https://github.com/user-attachments/assets/70aa39fb-c7a8-4bf7-8b85-32a187b5e706)

From the histograms above, we observe distinct patterns in the distribution of passenger ratings and flight-related attributes. Age is approximately symmetric, with most passengers in the 20–50 range. Flight distance is highly skewed to the right, which indicates most passengers travel short distances, with few taking long flights. Service-related features like inflight WiFi, food and drink, and online boarding show slight right skews, which indicate moderate to high satisfaction. On the other hand, ratings for onboard service, inflight entertainment, and baggage handling are slightly skewed to the left, which reflects generally high satisfaction in these areas. Gate location and seat comfort exhibit relatively uniform distributions. Departure and arrival delays are highly skewed to the right with the majority of passengers experiencing minimal delays. These patterns reveal a positive trend in passenger satisfaction for most service attributes while highlighting challenges with delays and flight distance distribution.

### Correlation Heatmap
![image](https://github.com/user-attachments/assets/cbe9e89a-c86b-4b2e-863a-59d70a2f65b2)

One of the strongest positive correlations is observed between Departure Delay in minutes and Arrival in Minutes (0.97). This indicates that departure delays directly influence arrival delays. This also emphasizes the need for airlines to address departure punctuality to improve overall timeliness and customer satisfaction. To avoid multicollinearity issues, we decided to drop variables that have any correlation above 0.8 or below -0.8 thresholds, which includes arrival and departure delay in minutes. Both have similar low correlations to other variables, so we randomly chose arrival delay in minutes to be dropped.

## Model Benchmarking and Optimization
### Naive Bayes
#### Benchmark Model
As a benchmark model, we used sklearn’s Gaussian Classifier to create a Naive Bayes model. Besides removing the 393 missing rows of data (as stated in the data cleaning section), we implemented this program without processing any of the data. The following is our benchmark Naive Bayes classifier model: 

![image](https://github.com/user-attachments/assets/867f2380-7b7f-46bd-8b6d-c8e56f0e86c2)

The benchmark model has a training set accuracy of 60.09% which shows no signs of overfitting. However, the overall model accuracy is only 60.06% which is a very weak machine learning model. Multiple data processing steps can be used to improve on this benchmark. 



### Decision Tree


### Random Forest

## Model Interpretation


## Final Remarks









