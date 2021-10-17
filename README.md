# House Price Prediction Analysis
## Machine Learning Model Development

### About -
This Project here deals with a fictional real estate company called "Dragon Real Estates". They are struggling in terms of predicting the prices of the houses in a specific area they want to expand in. Hence they require an accurate predictive analysis so that their buying decisions doesn't turn out unreasonable.


### Understanding the requirement -
As Machine learning engineers our first job is to understand the requirements properly. Exploring the business objectives . Getting an idea of what resources available and sharing what additional resources might be required for the project. These meeting are to be conducted along with stakeholders of the company. We also need to understand that what kind of solution is currently being implemented to address this issue.

### Understanding the Dataset -
The Dataset shared with us need to be analysed in-depth to provide an acurate model. currently the data set we have consist of the following details -

#### 1. Title: Boston Housing Data

#### 2. Sources:
   (a) Origin:  This dataset was taken from the StatLib library which is
                maintained at Carnegie Mellon University.
   (b) Creator:  Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the 
                 demand for clean air', J. Environ. Economics & Management,
                 vol.5, 81-102, 1978.
   (c) Date: July 7, 1993

#### 3. Past Usage:
   -   Used in Belsley, Kuh & Welsch, 'Regression diagnostics ...', Wiley, 
       1980 N.B. Various transformations are used in the table on
       pages 244-261.
    -  Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning.
       In Proceedings on the Tenth International Conference of Machine 
       Learning, 236-243, University of Massachusetts, Amherst. Morgan
       Kaufmann.

#### 4. Relevant Information: Concerns housing values in suburbs of Boston.
#### 5. Number of Instances: 506
#### 6. Number of Attributes: 13 continuous attributes (including "class" attribute "MEDV"), 1 binary-valued attribute.
#### 7. Attribute Information:

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town
    13. LSTAT    % lower status of the population
    14. MEDV     Median value of owner-occupied homes in $1000's

### Model Selection -
After understanding the dataset completely , we need to select a few models that we want to use to train our data on.

In here our target variable is ' MEDV '  which is the house prices and it's a continious variable. So initially we'll use Linear Regression for training and later on , to improve accuracy we'll check if decision tree or random forest works better with our data.

This model will be a supervised learning as we have labels of the data with us.

As in this dataset we have data that is already present with us and no live streaming data so we will go for batch learning instead of online learning.

### Selection Of Performence Measure -
As the problem we are working with is a regression problem hence we will use RMSE(root mean squared error). This will reduce the error margin. As we get both positive and negative error in out dataset where the data points are on both side of the best fit line , the squared value makes it a positive one.

## Workflow -

* We will use Jupyter Notebook for analysis.
* We will use visual studio Code for deployment.
* Libraries used -
       > Jupyter notebook.
       > Pandas.
       > Numpy.
       > ScikitLearn.
       > Joblib.
       > Pickle.

1. Data Analysis -
       * Understanding the features available.
       * Missing Value handling.
2. Train test Split.
3. Visualization on training data.
4. Models Selection.
5. Best model choosing.
6. Checking the score with test data.
7. Dumping the model.






