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
- [Model Comparison and Interpretation](#model-comparison-and-interpretation)
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


#### Optimiziation Method 1 ~ Wrapper Method Feature Selection 
One strategy for improving the accuracy of a machine learning model is to use a Feature Selection tool to find the optimal number of attributes that will deliver the highest accuracy. For this part, we used 10-Fold Cross Validation on the training data and found the mean accuracy for different features selected.

![image](https://github.com/user-attachments/assets/b4cb61b5-2235-494c-9c90-b312793927c3)

This resulted in an optimal model that includes 10 features selected from the dataset. The following table shows our results when testing for the model’s accuracy with the specified features:

![image](https://github.com/user-attachments/assets/6bf73bab-d614-4e3a-911e-16b9b673969d)

10 Features Used: Gender, Customer Type, Age, Type of Travel, Inflight wifi service, Departure/Arrival time convenient, Gate location, Baggage handling, Business, Eco Plus


#### Optimiziation Method 2 ~ Binning Continuous Variables
The next pre-processing step we used to improve accuracy to our model was categorizing any continuous variables. The Naive Bayes algorithm works best when attributes are in discrete categories rather than attributes that have continuous values. For this step, I separated the data into bins to create a more normal distribution shape. The 4 continuous variables we categorized are ‘Age’, ‘Flight Distance’, ‘Departure Delay in Minutes’, and ‘Arrival Delay in Minutes’. The following graphs show the new distribution of the binned variables:

![image](https://github.com/user-attachments/assets/d73d85c5-7cbd-47b1-97d9-c8a11bd00273)
![image](https://github.com/user-attachments/assets/7f75b1c1-7404-4dfe-b4f0-76a8efd9c7b6)
![image](https://github.com/user-attachments/assets/3a2aad07-19db-4981-b480-6f0e201fd272)
![image](https://github.com/user-attachments/assets/24395482-01de-4ba0-acc5-b7bb41d49b0e)

Below are the results of our binned model which has improved from our benchmark model and scored similarly to our feature selection model. The overall test score implies that this model classifies customer satisfaction levels correctly 75.97% of the time. In our last pre-processing technique, we will combine the methods of binning and feature selection to see if we can increase the accuracy more

![image](https://github.com/user-attachments/assets/3ac9a6b7-ac94-4b4b-8e85-b7841cccb55c)


#### Optimiziation Method 3 ~ Feature Selection on Binned Data
The final optimization method is a combination  of the previous two, where we performed wrapper method feature selection on the dataset with the binned continuous variables. The following table shows the 10 fold cross validation accuracy to its corresponding numbers of features selected and the naive bayes model performance based on the optimal number of features that should be selected (highlighted in yellow). 

![image](https://github.com/user-attachments/assets/c8934432-0c66-4760-8582-77cd8de75c9c)


![image](https://github.com/user-attachments/assets/ff145e03-e2de-4d2c-8c67-38c85b9571c1)

*5 Features Used: Gender, Inflight wifi service, Departure/Arrival time convenient, Gate location, Business

By using feature selection the binned dataset, we were able to further improve the accuracy of our Naive Bayes model. This model also performed moderately well in the recall section, showing that it will correctly predict ‘Not Satisfied/Neutral’ 80% of the time and ‘Satisfied’ 77% of the time.


#### Naive Bayes Model Comparison
![image](https://github.com/user-attachments/assets/748a0b33-6da4-437f-a449-215290f558a6)

After analyzing the three types of pre-processing techniques, the model that performed the best was our latest model. This is mainly due to Naive Bayes classifiers being able to handle categorical variables more effectively than continuous variables. Along with this, we used 10-fold cross validation feature selection to find the most important features and filter out any attributes causing unnecessary noise in the data. In the next section of our report, we use Random Forest and Decision Trees in an attempt to improve accuracy and recall.


### Decision Tree
As a benchmark model, we simply used sklearn’s DecisionTreeClassifier function to create an unprepossessed decision tree model with no edited parameters aside from entropy as the model’s criterion. Any correlating variables (arrival delay time) have also been removed to avoid multicollinearity in this model. The performance of the tree is as follows. 

![image](https://github.com/user-attachments/assets/0a7e7c75-8ac2-43d9-a76c-b5cf957e3656)

As seen by the model’s performance, this model is somewhat overfitted. It has a 100% accuracy on the training data, but only a 94.81% accuracy on the testing data. The precision, recall, and f1-score are also equal between the not satisfied/neutral class and satisfied, each bearing 95% accuracy and 94% accuracy respectively per class. Pre-pruning and post-pruning on the trees are performed to remedy the overfitting issue.


#### Optimiziation Method 1 ~ Pre-pruning using Hyperparameter Tuning Using Gridsearch 
The first method we used to remove overfitting in our model is by pre pruning or limiting the size of the tree’s growth. This involves specifying the maximum depth, minimum sample splits, and minimum sample leaves of the tree. To find the optimal combinations of these three parameters, our group utilized hyperparameter tuning using grid search from the scikit-learn package. This algorithm runs through all combinations of the specified parameters and uses cross validation to identify the best combination based on model performance.  

We performed hyperparameter tuning three times based on different parameter values to determine the best parameter combination to use for the pre-pruned decision tree model. The following table describes each of the three trial’s parameter ranges, selected parameters from grid search, and their corresponding 10-fold cross validation accuracy and testing accuracy. 

![image](https://github.com/user-attachments/assets/dd9c2089-54f6-4c76-9e77-13366391000d)

We started with very broad parameter ranges on the first trial (ranging from 10 to 100), to identify a rough location for the optimal parameters. On the second trial, we dug deeper into more accurate ranges. For example, since the optimal value for ‘min samples split’ was at 100 in trial 1, we chose a range between 80 and 120 in intervals of 10 to find a more optimal value close to 100. On the third try, the optimal value for max depth is 21, which concluded that the best max depth for a tree with 100 minimum samples split is between 21 and 22 from the second and third trial. The minimum sample leaf continued to decrease 10 to 5 and to 3, across the three trials. However, we didn’t want our tree’s leaf nodes to have a small sample size, so we decided to stick with three and not continue decreasing the parameter ranges. 

![image](https://github.com/user-attachments/assets/0532142e-0df2-4928-ab6f-c6ec490fc74f)

According to the table above, the best model is the trial 3 model, with the highest 10 fold cross validation accuracy of 95.42%. We then used the parameters from this trial 3 model to build a new decision tree model, which we tested on the actual training and testing datasets. The parameters we used and the results of our optimal pre pruned decision tree model is described below. 


![image](https://github.com/user-attachments/assets/874fc3e2-5a29-49be-8496-a664bf6e9fbd)


![image](https://github.com/user-attachments/assets/9760697b-4f8b-4a1c-8176-97285c16f0b7)


Compared to the benchmark decision tree model, our optimal pre-pruned model is no longer overfitted. The benchmark model had a training accuracy of 100% and a testing accuracy of 94.81%, showing an accuracy discrepancy of more than 5%. However, the accuracy discrepancy was less than 1% in our pre-pruned model, as the training accuracy is 96.20% and the testing accuracy is 95.56%. So in regard to testing and training accuracy, this optimal pre pruned model performs better than the benchmark decision tree model. 

However, based on the recall, this pre pruned model correctly predicts ‘Not Satisfied or Neutral’ classes 97% of the time, but only 93% of the time for ‘Satisfied’ classes. Clearly, this model is slightly better at predicting ‘Not Satisfied or Neutral’ classes than ‘Satisfied’ A potential cause is the dataset’s class imbalance, as ‘Not Satisfied or Neutral’ accounts for 56.7% and ‘Satisfied’ only 43.3% of the dependent variable. The SMOTE section of this report will try to remedy this issue. 


#### Optimiziation Method 2 ~ Post-pruning Using Cost Complexity Pruning Algorithm  

Aside from pre-pruning, another method to prevent a decision tree from overfitting is through post-pruning. Unlike pre-pruning which involves parameters like maximum tree depth and minimum sample splits, the post-pruning cost complexity method finds the best value for alpha by comparing the full grown tree’s accuracy with a corresponding alpha value. The method to find the optimal alpha is by graphing out the accuracy vs alpha chart and selecting the alpha value where the test accuracy is at its highest or before it starts decreasing. This method will balance the models’ predictive performance and complexity. 

The graph below shows our decision tree’s accuracy against alpha. Accuracy is on the y-axis and alpha on the x-axis. The optimal test accuracy point before decreasing is somewhere close to 0.002 alpha. Hence, we will choose a 0.2% alpha as our optimal alpha parameter. 

![image](https://github.com/user-attachments/assets/2c6e5c36-e804-4aef-aa77-deb5d844568f)

After running a decision tree model and specifying the ccp_alpha equal to 0.2%, we got the following model performance. 

![image](https://github.com/user-attachments/assets/9d85b12e-5df2-44b5-b5e4-05b901d8f9f8)

This post-pruned model showed no signs of overfitting as the training and testing accuracy are very similar at 93.62% and 93.44%, respectively. This shows that the cost complexity pruning method is effective in reducing overfitting for this airline satisfaction dataset. However, in comparison to the pre-pruning model, this model slightly pales in comparison in regards to predictive capabilities. The test accuracy for our pre-pruned model is around 95.56%, which is 2.12% higher than this post-pruned model. One thing that post and pre pruned models have in common is their tendency to predict ‘Not Satisfied / Neutral’ classes better than ‘Satisfied’ classes, reflected in their recall accuracy.


#### Optimiziation Method 3 ~ SMOTE Balancing Using Optimal Pre Pruning Parameters

![image](https://github.com/user-attachments/assets/d04c11f5-9e17-48a7-b430-15ec3eaddba8)

All models face the same condition of classifying “Not Satisfied/Neutral” (0) classes slightly better than “Satisfied” (1) classes. One reason for this is the dependent variable’s class imbalance. The following pie chart shows the proportion of 1 and 0 in the full dataset and after partitioning into training and testing datasets. 

![image](https://github.com/user-attachments/assets/cb157b62-31be-4a39-beaa-37164156af7c)
![image](https://github.com/user-attachments/assets/884205e8-46fe-45ca-993c-1d626a839103)
![image](https://github.com/user-attachments/assets/8fe0cad0-c342-4d74-9e0c-628f4b3641c6)

To fix the class imbalance, we used the SMOTE algorithm on the training dataset to create equal proportions of 1 and 0 in the dependent variable. Prior to SMOTE, the count of 1 and 0 in the training data is 39,383 and 51,257 respectively. After SMOTE, the count of 1 and 0 becomes 51,257 and 51,257. The SMOTE algorithm oversamples the “Satisfied (1)” classes to match the number of “Not Satisfied/Neutral (0)” classes in the training dataset. Since the pre-pruned model has the highest overall accuracy, we created a decision tree model with this new SMOTE training dataset while using our pre-pruned model’s optimal gridsearch parameters. 

![image](https://github.com/user-attachments/assets/b35d56e8-8bbc-4f77-8791-e40b4d8e6dd6)

This SMOTE pre-pruned decision tree model has a better capability of classifying 1 and 0 more evenly compared to other models. As seen by the accuracy statistics values above, the recall accuracy for both class 1 and 0 are equal at 95%, with close to 0% discrepancy between the two accuracies. The post pruned and pre pruned models have a discrepancy of around 4%, so the SMOTE technique improved the class imbalance performance by around 4%. However, overall testing accuracy decreased in this pre pruned model due to the SMOTE. The table below shows the differences between the pre pruned models before and after SMOTE. 

![image](https://github.com/user-attachments/assets/1907f075-6ffb-49fb-8732-7ec06e972427)

The table shows that to get an equal recall accuracy for both classes, the overall model’s testing accuracy will decrease by 0.46%. However, even if that’s the case, the SMOTE pre-pruned model still has a decent 95.10% testing accuracy. The following section will compare the accuracies from all decision trees explained above and determine which decision tree is the best. 


#### Decision Tree Model Comparison

![image](https://github.com/user-attachments/assets/470871c2-9e21-4daf-b4b2-bfb7916791dd)

By comparing the overall testing accuracy, the pre-pruned model (non SMOTE) and the SMOTE pre pruned model have the two highest performance. The benchmark model is overfitted, which makes this model unusable and the post pruned model has the lowest testing accuracy compared to the other three models. The two pre-pruned models have very similar testing accuracy, with a discrepancy of less than 0.5%. Because of this, choosing which of these two is the best model depends on the goal and scope of the project. We believe an airline must correctly classify customers who are neutral or dissatisfied with its service. This way, the airline can provide tailored marketing of sorts to transform these ‘class 0’ customers to satisfied customers. However, it is equally important for an airline to classify satisfied customers correctly as to provide tailored marketing to keep these customers as satisfied. From this logic, classifying ‘class 0’ customers correctly is equally important as classifying ‘class 1’ customers correctly. Hence, we believe that the SMOTE + pre-pruned model is the best model in the decision tree category. 


### Random Forest
Another model we used to predict customer satisfaction for our airline dataset was using a Random Forest classifier. For our dataset, this model is an improvement from the Naive Bayes classifier as it is more robust when it comes to handling missing values and continuous variables. The following image is our benchmark model which will be used as a base:

![image](https://github.com/user-attachments/assets/65aa6dec-bdd9-40a2-a167-6a5e2d7f136f)

Our Random Forest benchmark model shows a training set accuracy of 99.99% (1 error) and a testing set accuracy of 96.13% (over 1000 errors). This is an indication that our benchmark model is overfitted and too highly trained to the specific training examples. Having an overfitted model will result in poor generalized performance so pre-processing techniques must be executed in order to account for overfitting in the model. 


#### Optimiziation Method 1 ~ Adjusting the Subset of Attributes at Each Split
In the first processing technique, we tried limiting the number of attributes available at each split in the Random Forest model. The purpose of this is to increase diversity in each Decision Trees to create different results for the Random Forest algorithm. The table below shows our results when changing the ‘max feature’ parameter. However, in all of these trials we have a model that is overfitted, making it no more reliable than our benchmark model. 

![image](https://github.com/user-attachments/assets/2e3b366c-3a23-4131-b10b-2b2b1a50eea0)


#### Optimiziation Method 2 ~ Pre-pruning Using Hyperparameter Tuning (Grid Search)
One main way to fix overfitting is to pre-prune the Random Forest classifier. This limits the size of each tree by putting restrictions on different hyperparameters. In our model we restricted the max depth of the tree, put a minimum number of samples on a node for it to split, created a minimum threshold for each leaf node, and created different subset of attributes. To find the optimal combination of these parameters, we used sklearn’s GridSearchCV program and 10-fold cross validation to find the optimal parameter values.

For the purpose of this project, we limited the number of parameters and ran multiple trials to save time and computing power. We also limited the number of trees in the Random Forest to 5. The following table shows each iteration from our trials:

![image](https://github.com/user-attachments/assets/e8ceaac2-6267-484b-9005-69c82499b1ad)

On our first trial, we used a wide range of parameter values to estimate where the optimal parameter would be. Once the Grid Search returned the selected option, we ran another iteration within a range closer to the selected option. We repeated this through our four trials to close in on best parameter values. We found the optimal parameter combination for our Random Forest model is a max depth of 21, minimum sample split of 37, minimum sample leaf of 10, and no max feature subset.

The following performance chart presents the outcome of the Random Forest model when using 200 decision trees and the identified optimal hyperparameters.

![image](https://github.com/user-attachments/assets/f68c8816-dc8d-46ae-9259-01dcfe76d1d8)

Even though the 10-fold cross validation mean accuracy is lower than the benchmark model, the reliability of our new Random Forest model exceeds that of the benchmark model since we have solved for overfitting. This model consists of the parameters found in the 4th GridSearchCV trial to create a Random Forest model that is not overfitted with a training accuracy of 96.79% and a testing accuracy of 96.10%. Compared to the benchmark model, the accuracy has decreased slightly from 96.13% to 96.10% but we have accounted for the overfitted model thus creating a Random Forest model that has a higher generalized performance. 


#### Random Forest Model Comparison
![image](https://github.com/user-attachments/assets/0e1fd12c-5ba6-4ad9-883b-84790e6e93e3)

After multiple data processing techniques, the Random Forest Model that performed the best was when we used 10-fold cross validation and GridSearchCV to identify the optimized parameters for pre-pruning. The table below shows the top five most important features in predicting customer satisfaction levels in our optimal Random Forest model. The first three are substantially more influential than the rest, showing that airline managers should put greater value on these attributes than others.

![image](https://github.com/user-attachments/assets/e6137475-60fd-4e2a-b293-afd1e20c05fe)


## Model Comparison and Interpretation
#### Model Comparison

![image](https://github.com/user-attachments/assets/9192e293-0aa1-4ed7-871c-ce8495d7f412)

Model testing accuracy and interpretability are the two criteria we use to choose the best model for this project. Our goal is to provide airlines with an accurate model that is capable of providing valuable insights on what factors impact customer satisfaction so that airlines can create solutions based on these factors. In regards to testing accuracy, naive bayes will not be considered as our best model since it has a 20% lower accuracy than our random forest and decision tree model. In regard to model interpretation, decision trees are more comprehensible to readers than random forest, since readers can tell which variables and thresholds are used in each node in the decision tree to create purer subdivisions of the classes. Hence, although random forest is slightly more accurate, the decision tree model is the best model to be used for the scope of this project.


#### Decision Tree Model Implications 

![image](https://github.com/user-attachments/assets/a68e6a3c-d80e-44c3-87d6-bc70280a0fe2)

Because python stacks the nodes on top of each other which makes the tree difficult to read, the following graph was made to display the top nodes. Typically, the top nodes are the most important features because they perform the largest splits and have the highest information gain. Hence, the following graph displays the top root nodes of the decision tree which represents the variables that are the most important in classifying satisfied from non satisfied/neutral customers. 

![image](https://github.com/user-attachments/assets/cc15cf2f-780c-4802-9176-5c1088d1dca8)

The most important attribute is online boarding, found in node 1. If online boarding rating (ranging from 0 to 5) is below 3.5, the root node will continue left to node number 2, where the majority class in this node is “Not Satisfied/Neutral.” However, if node 1’s online boarding rating is above 3.5, the root node will continue right to node 3, where the majority of the class is “Satisfied.” 

Continuing from node number 2, if the rating of inflight WiFi service is greater than 0.5, the tree will continue to node number 4 where the majority of the class is “Not Satisfied/Neutral.”  From node 4, if the rating of inflight WiFi service is greater than 3.5, the tree will continue to node 7 where majority of the class is “Satisfied.”

Continuing from node 3, if business travel is less than 0.5, the tree will continue to node 5 where the majority of the class is “Not Satisfied/Neutral.” If business travel is greater than 0.5, the tree will continue to node 6 where the majority of the class is “Satisfied.” The variable business travel is a categorical variable where personal travel is denoted as 0 and business travel denoted as 1. So from the tree analysis, personal travel tends to lead to “Not Satisfied/Neutral” while business travel leads to “Satisfied.”


## Final Remarks

Key airline service areas from decision tree:
Online boarding service
- Above 3.5 rating leads to majority satisfied 
- Below 3.5 rating leads to majority not satisfied or neutral

Inflight WiFi service 
- Above 0.5 rating still leads to majority not satisfied or neutral
- Above 3.5 rating leads to majority satisfied (3.5 rating is the a good benchmark to create majority satisfied customers)

Type/Purpose of Travel
- Business travel leads to majority satisfied 
- Personal travel leads to majority not satisfied or neutral

We highly recommend airlines to optimize the three service areas above to retain existing customer satisfaction and transform not satisfied or neutral customers to become satisfied. These three service areas are at the top of the decision tree, which implies that they hold more significance in this classification model compared to the other variables present. Hence, focusing on optimizing and improving online boarding service, inflight WiFi service, and other services tailored to business and personal travellers is a solid next step for the airline companies. 









