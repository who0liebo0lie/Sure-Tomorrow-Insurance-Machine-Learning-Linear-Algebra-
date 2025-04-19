#!/usr/bin/env python
# coding: utf-8

# # Sure Tomorrow (Machine Learning Linear Algebra)

# The Sure Tomorrow insurance company provides a file for use of a project (file /datasets/insurance_us.csv).  Once data is loaded it will be checked for issues (there is no missing data, extreme values, and so on).  Listed below are tasks to be solved using Machine Learning. 
# 
# Features: insured person's gender, age, salary, and number of family members.
# Target: number of insurance benefits received by an insured person over the last five years.
# 

# # Statement

# The Sure Tomorrow insurance company wants to solve several tasks with the help of Machine Learning and you are asked to evaluate that possibility.
# 
# - Task 1: Find customers who are similar to a given customer. This will help the company's agents with marketing.
# - Task 2: Predict whether a new customer is likely to receive an insurance benefit. Can a prediction model do better than a dummy model?
# - Task 3: Predict the number of insurance benefits a new customer is likely to receive using a linear regression model.
# - Task 4: Protect clients' personal data without breaking the model from the previous task. It's necessary to develop a data transformation algorithm that would make it hard to recover personal information if the data fell into the wrong hands. This is called data masking, or data obfuscation. But the data should be protected in such a way that the quality of machine learning models doesn't suffer. You don't need to pick the best model, just prove that the algorithm works correctly.

# # Data Preprocessing & Exploration
# 
# ## Initialization

# In[1]:


import numpy as np
import pandas as pd
import math

#import named regression models 
from sklearn.linear_model import LinearRegression

import seaborn as sns

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score


from IPython.display import display

from sklearn.preprocessing import StandardScaler




# ## Load Data

# Load data and conduct a basic check that it's free from obvious issues.

# In[2]:


df = pd.read_csv('/datasets/insurance_us.csv')


# We rename the colums to make the code look more consistent with its style.

# In[3]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})


# In[4]:


df.sample(10)


# In[5]:


df.info()


# In[6]:


# we may want to fix the age type (from float to int) though this is not critical
df['age']=df['age'].astype(int)


# In[7]:


# check to see that the conversion was successful
df.info()


# In[8]:


# now have a look at the data's descriptive statistics. 
df.isna().sum()


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# Count shows that there is no missing features (which we realized from checking above). Noticable that the youngest age is 8.44. 

# ## EDA

# Let's quickly check whether there are certain groups of customers by looking at the pair plot.

# In[11]:


g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


# It is difficult to spot obvious groups (clusters) as it is difficult to combine several variables simultaneously (to analyze multivariate distributions). 

# # Task 1. Similar Customers

# Develop a procedure that returns k nearest neighbors (objects) for a given object based on the distance between the objects.

# Write function that returns k nearest neighbors for an $n^{th}$ object based on a specified distance metric. The number of received insurance benefits should not be taken into account for this task. 
# 
# Implementation of the kNN algorithm from scikit-learn (check [the link](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)) or use your own.
# 
# Test it for four combination of two cases
# - Scaling
#   - the data is not scaled
#   - the data is scaled with the [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) scaler
# - Distance Metrics
#   - Euclidean
#   - Manhattan
# 

# In[12]:


feature_names = ['gender', 'age', 'income', 'family_members']


# In[13]:


def get_knn(df, n, k, metric):
    
    """
    Returns k nearest neighbors

    :param df: pandas DataFrame used to find similar objects within
    :param n: object no for which the nearest neighbours are looked for
    :param k: the number of the nearest neighbours to return
    :param metric: name of distance metric
    """
    # Instantiate the NearestNeighbors object with k, metric
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
    
    # Fit the model with the DataFrame excluding the target/index n
    nbrs.fit(df[feature_names])
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res


# Scaling the data.

# In[14]:


feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())


# In[15]:


df_scaled.sample(5)


# Now, let's get similar records for a given one for every combination

# In[16]:


# List of distance metrics
metrics = ['euclidean', 'manhattan']

# Define the number of nearest neighbors
k = 8

# Define the sample customer index 
n = 3

# Original data (non-scaled)
print("Finding similar records (non-scaled):")
for metric in metrics:
    similar_records = get_knn(df, n, k, metric)
    print(f"\nUsing metric: {metric}")
    print(similar_records)

# Scaled data
print("\nFinding similar records (scaled):")
for metric in metrics:
    similar_records = get_knn(df_scaled, n, k, metric)
    print(f"\nUsing metric: {metric}")
    print(similar_records)


# In[17]:


# Define the number of nearest neighbors
k = 10

# Define the sample customer index 
n = 0

# Original data (non-scaled)
print("Finding similar records (non-scaled):")
for metric in metrics:
    similar_records = get_knn(df, n, k, metric)
    print(f"\nUsing metric: {metric}")
    print(similar_records)

# Scaled data
print("\nFinding similar records (scaled):")
for metric in metrics:
    similar_records = get_knn(df_scaled, n, k, metric)
    print(f"\nUsing metric: {metric}")
    print(similar_records)


# **Does the data being not scaled affect the kNN algorithm?** 
# 
# Unscaled data significantly affects the kNN algorithm. Features with larger numeric ranges (like income) dominate the distance calculations.  This forces features like age and gender with smaller numerical ranges to become miniscule. This kNN algorithm  relying on the feature with the largest range can bias nearest neighbor results.
# 
# Scaling data forces each feature to have equal contributions to the distance calculation. The kNN algorithm is able to find more accurate nearest neighbors. This leads to more reasonable predictions and better representation of the actual similarity between the data points.

# **How similar are the results using the Manhattan distance metric (regardless of the scaling)?** 
# 
# The Manhattan distance is consistenlty less sensitive to outliers since it calculates distances as the sum of absolute differences. Regardless of scaling the distance increases incrementally for each row. 
# 

# # Task 2. Is Customer Likely to Receive Insurance Benefit?

# Binary Classification Task:
# With `insurance_benefits` being more than zero as the target, evaluate whether the kNN classification approach can do better than a dummy model.
# 
# - Build a KNN-based classifier and measure its quality with the F1 metric for k=1..10 for both the original data and the scaled one. That'd be interesting to see how k may influece the evaluation metric, and whether scaling the data makes any difference. You can use a ready implemention of the kNN classification algorithm from scikit-learn (check [the link](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) or use your own.
# - Build the dummy model which is just random for this case. It should return "1" with some probability. Let's test the model with four probability values: 0, the probability of paying any insurance benefit, 0.5, 1.
# 
# The probability of paying any insurance benefit can be defined as
# 
# $$
# P\{\text{insurance benefit received}\}=\frac{\text{number of clients received any insurance benefit}}{\text{total number of clients}}.
# $$
# 
# Split the whole data in the 70:30 proportion for the training/testing parts.

# In[18]:


# calculate the target
df['insurance_benefits_received'] = (df['insurance_benefits']>0).astype(int)


# In[19]:


# check for the class imbalance with value_counts()
df['insurance_benefits_received'].value_counts()


# In[20]:


X = df[['gender', 'age', 'income', 'family_members']]
y = df['insurance_benefits_received']


# In[21]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[22]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled= train_test_split(X_scaled, y, test_size=0.3, random_state=42)
   
f1_scores_original = []
f1_scores_scaled = []
   
for k in range(1, 11):
    # Original data
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    f1_scores_original.append(f1_score(y_test, y_pred))
   
    # Scaled data
    knn.fit(X_train_scaled, y_train)
    y_pred_scaled = knn.predict(X_test_scaled)
    f1_scores_scaled.append(f1_score(y_test, y_pred_scaled))
   
print("F1 Scores for Original Data: ", f1_scores_original)
print("F1 Scores for Scaled Data: ", f1_scores_scaled)
   


# In[23]:


import matplotlib.pyplot as plt
   
plt.plot(range(1, 11), f1_scores_original, label='Original Data')
plt.plot(range(1, 11), f1_scores_scaled, label='Scaled Data', linestyle='--')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Number of Neighbors')
plt.legend()
plt.show()
   


# In[24]:


def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')
    
# if you have an issue with the following line, restart the kernel and run the notebook again
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print('Confusion Matrix')
    print(cm)


# In[25]:


# generating output of a random model

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# In[26]:


for P in [0, df['insurance_benefits_received'].sum() / len(df), 0.5, 1]:

    print(f'The probability: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, len(df))
        
    eval_classifier(df['insurance_benefits_received'], y_pred_rnd)
    
    print()


# # Task 3. Regression (with Linear Regression)

# Evaluate what RMSE would be for a Linear Regression model.

# Check RMSE for both the original data and the scaled one; spot differences.
# 
# Let's denote
# - $X$ — feature matrix, each row is a case, each column is a feature, the first column consists of unities
# - $y$ — target (a vector)
# - $\hat{y}$ — estimated tagret (a vector)
# - $w$ — weight vector
# 
# The task of linear regression in the language of matrices can be formulated as
# 
# $$
# y = Xw
# $$
# 
# The training objective then is to find such $w$ that it would minimize the L2-distance (MSE) between $Xw$ and $y$:
# 
# $$
# \min_w d_2(Xw, y) \quad \text{or} \quad \min_w \text{MSE}(Xw, y)
# $$
# 
# It appears that there is analytical solution for the above:
# 
# $$
# w = (X^T X)^{-1} X^T y
# $$
# 
# The formula above can be used to find the weights $w$ and the latter can be used to calculate predicted values
# 
# $$
# \hat{y} = X_{val}w
# $$

# Split the whole data in the 70:30 proportion for the training/validation parts. Use the RMSE metric for the model evaluation.

# In[27]:


class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None
    
    def fit(self, X, y):
        
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T @ X2) @ X2.T @ y


    def predict(self, X):
        
        # adding the unities
        X2 = np.append(np.ones((len(X), 1)), X, axis=1)
        y_pred = X2 @ self.weights
        
        return y_pred


# In[28]:


def eval_regressor(y_true, y_pred):
    
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    


# In[29]:


X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print('Original Linear Regression Results:')
print(lr.weights)

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# In[30]:


X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled= train_test_split(X_scaled, y, test_size=0.3, random_state=12345)


# In[31]:


print('Scaled Linear Regression Results:')
lr.fit(X_train_scaled, y_train_scaled)
print(lr.weights)

y_test_pred = lr.predict(X_test_scaled)
eval_regressor(y_test_scaled, y_test_pred)


# # Task 4. Obfuscating Data

# It best to obfuscate data by multiplying the numerical features (remember, they can be seen as the matrix $X$) by an invertible matrix $P$. 
# 
# $$
# X' = X \times P
# $$
# 
# Check how the features' values will look like after the transformation. Intertible property is important here so make sure that $P$ is indeed invertible.
# 

# In[32]:


personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# In[33]:


X = df_pn.to_numpy()


# Generating a random matrix $P$.

# In[34]:


rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))


# In[35]:


P


# Checking the matrix $P$ is invertible

# In[36]:


inverse=np.linalg.inv(P)
print(inverse)


# Matrix P is invertible. 

# Recover the original data from $X'$ if $P$ is known? Try to check that with calculations by moving $P$ from the right side of the formula above to the left one. The rules of matrix multiplcation are really helpful here.

# In[37]:


inverse/P


# Print all three cases for a few customers
# - The original data
# - The transformed one
# - The reversed (recovered) one

# In[38]:


# Transform the original data
transformed = np.dot(X, P)

# Compute the inverse of P
P_inverse = np.linalg.inv(P)

# Recover the original data
recovered = np.dot(transformed, P_inverse)

# Print results for demonstration purposes
# (You can customize this to print a few customers)
print("Original Data:")
print(X)

print("\nTransformed Data:")
print(transformed)

print("\nRecovered Data:")
print(recovered)


# Some values may n ot be exactly the same due to numpy only functioning with double precision.  Project matrixes have values at e-13.  Rounding errors can still occur. 

# ## Proof That Data Obfuscation Can Work with LR

# The regression task has been solved with linear regression in this project. Your next task is to prove _analytically_ that the given obfuscation method won't affect linear regression in terms of predicted values i.e. their values will remain the same.

# If formula was simpliefied $w$ and $w_P$ would be linked as $w_P$= P^-1w.  The predicted values are the same as using w.  The obfuscation through 𝑃 does not alter the predictions. RMSE the discrepeancy between the true values and the predicted values.  The quality of linear regression measured with RMSE remains unchanged since the predicted values $y_P$ with $w_P$ identical to y with w. 

# **Analytical proof**

# Given that 
# $$w = (X^TX)^{-1}X^Ty$$
# $$w_P = [(XP)^T(XP)]^{-1}(XP)^Ty$$
# 1. Based on reversaity of transpose of matrix product on first half of equation: 
# $$ w_P = [(XP)^TXP]^{-1} (XP)^Ty$$
# 2. associative property of multiplication
# $$ w_P = [(P^TX^TX)P]^{-1} (XP)^Ty$$
# 3. multiplicative identity property
# $$ w_P = P^{-1}[P^TX^TX]^{-1} (XP)^Ty$$
# 4. multiplicative identity property
# $$ w_P = P^{-1}[X^TX]^{-1}[P^T]^{-1} (XP)^Ty$$
# 5. Based on reversaity of transpose of matrix product on second half of equation: 
# $$ w_P = P^{-1}[X^TX]^{-1}[P^T]^{-1} (P^TX^Ty)$$
# 6. Identify and remove present identity matrix for P
# $$ w_P = P^{-1}[X^TX]^{-1}IX^Ty$$
# $$ w_P = P^{-1}[X^TX]^{-1}X^Ty$$
# 7. Use w since definition present
# $$ w_P = P^{-1} w$$

# Linear regression is y=X(features) *w (weights)
# $$Y=XW$$
# $$yP=XP*wp$$
# insert in found formula for wp
# $$yP=XPP^{-1}w$$
# associative property of communication 
# $$yp=X*(PP^{-1})w $$
# remove identity matrix 
# $$yP=Xw$$
# There will not be a different result because if P is inverse we will achieve same result. 
# 
# 

# ## Test Linear Regression With Data Obfuscation

# Prove Linear Regression can work computationally with the chosen obfuscation transformation.
# 
# Build a procedure or a class that runs Linear Regression optionally with the obfuscation. 
# 
# Run Linear Regression for the original data and the obfuscated one, compare the predicted values and the RMSE, $R^2$ metric values. Is there any difference?

# In[39]:


rng = np.random.default_rng(seed=50)
P = rng.random(size=(X.shape[1], X.shape[1]))
inverse=np.linalg.inv(P)
print(inverse)


# In[40]:


# Linear regression function with option to obfuscate
def linear_regression(X, y, P=None):
    if P is not None:
        X = np.dot(X, P)
        
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return lr, y_pred, rmse, r2

# Splitting original data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Original Dataset: Train the model
lr_original, y_pred_original, rmse_original, r2_original = linear_regression(X_train, y_train)

# Use the trained model to make predictions on the test set
y_pred_test_original = lr_original.predict(X_test)
rmse_test_original = mean_squared_error(y_test, y_pred_test_original, squared=False)  
r2_test_original = r2_score(y_test, y_pred_test_original)

print("Original Dataset Evaluation")
print("RMSE:", rmse_test_original)
print("R2:", r2_test_original)


# Obfuscated Dataset: Train the model with obfuscated training data
lr_obfuscated, y_pred_obfuscated, rmse_obfuscated, r2_obfuscated = linear_regression(X_train, y_train, P)

# Obfuscate the test data using the same invertible matrix
X_test_obfuscated = np.dot(X_test, P)

# Use the trained model to make predictions on the obfuscated test set
y_pred_test_obfuscated = lr_obfuscated.predict(X_test_obfuscated)
rmse_test_obfuscated = mean_squared_error(y_test, y_pred_test_obfuscated, squared=False) 
r2_test_obfuscated = r2_score(y_test, y_pred_test_original)
print('******************************************')
print("Obfusciated Data Evaluation")
print("RMSE:", rmse_test_obfuscated)
print("R2:", r2_test_obfuscated)


# # Conclusions

# RMSE and ( R^2 ): The RMSE and ( R^2 ) values for both the original and obfuscated datasets are very similar. They only differ by a tenth decimal point.  Providing an indication that the obfuscation process did not significantly impact the performance of the Linear Regression. 
# 

# The transformation by an invertible matrix does not degrade the quality of the linear model.In using a random matrix for obfuscation the linear regression maintained the ability to make accurate predictions.

# Sensitive data can be obfuscated to protect it without sacrificing the accuracy of predictive models. Sure Tomorrow insurance company is enable to share data more securely.
# 
