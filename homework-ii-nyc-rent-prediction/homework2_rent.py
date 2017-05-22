
# coding: utf-8

# In[17]:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


# In[18]:

def build_model():
    
    """
    Parameters
    ----------

    None
    
    Returns
    -------
    
    grid : model (GridSearchCV object)
        GridSearchCV object fitted on the training dataset using MaxAbsScaler and Ridge regression.
    
    X_test : ndarray
        Numpy array holding the feature matrix for the test set.
    
    y_test : ndarray
        Numpy array holding values for response variable for the test set.
        
    Notes
    -----   
    Creates and builds a machine learning model to predict the monthly rent of an apartment using only features that apply to pricing 
    of an apartment that is not currently rented. 
    
    I. Assumptions:
    1. Market doesn't increase so the rent for a new tenant is the same as for the current tenant.
    2. The features selected does not include current tenants/ occupant details, expenditures, out of pocket rents etc.
    3. We have not included most of the continuous variables as they contain details related to occupied units and not vacant ones
    4. Recode and Flag variables have not been considered
    
    II. Feature selection and generation

    The data consists of 15342 rows and 197 columns. The columns are a mix of categorical and continuous variables.
    Not all of them influence the rent. After careful analysis, only 91 columns - 87 categorical and 3 continuous were chosen, which are 
    expected to influence the rent of a vacant apartment. The complete list of included variables is given in the .xls file.
    
    For all categorical variables, one hot encoding is performed. Missing/NA values are not imputed as their information is held by creating spearate binary columns for 
    each of the values. 
    
    'uf17' is chosen as the response variable y.
    
    All rows for which response variable y is missing or above topcode value, are dropped. The final dataset contains 10138 rows and 430 columns. 
    
    No imputation is necessary since the selected rows have no missing/NA values for the continuous variables.
    
    The dataset is then split into X(feature matrix) and y(response variable) and is split into a training and testing set in a 80:20 ratio.
    
    
    III. Model generation and selection
    
    
    Various linear models were tried on the dataset including Linear Regression, KNNRegressor, Lasso regression, Ridge regression, Elastic Net.
    Out of these Ridge regression gave the highest accuracy of 59.26% and was chosen to model this data.
    
    For training and modeling, pipelining was used to first scale the data using MaxAbsScaler. MaxAbsScaler was chosen since the data with its large number of binary columns has a large number of zero values and is sparse.
    GridSearchCV was then used to perform cross validation with 5 folds to determine the best alpha value for Ridge regression. The model with selected alpha value was then fit on the training dataset.
    
    """
    
    
    df = pd.read_csv('homework2_data.csv')

    non_cat_index = [1,32,35,36,52,54,56,58,62,72,73,82,84,85,87,89,91,92,99,141,142,143,144,145,147,149,151,153,155,157,159,161,164,165,168, 169,170]
    keep = list(range(2,31)) + [40,41,45,46,47,61,63,64] + list(range(66,82)) + [83,86,88,90]+ list(range(92,99)) + list(range(100,116)) + [118,126,127,128,129,130,137,138,139,140,163]
    
    col_names = list(df.columns.values)

    cat_names = []

    for i in range(0,197):
        if (i+1) not in non_cat_index:
            cat_names.append(col_names[i])

    

    keep_index = []

    for i in keep:
        keep_index.append(i-1)

    keep_df = df[keep_index]


    cat_names_new = []
    for name in cat_names:
        if name in list(keep_df.columns.values):
            cat_names_new.append(name)


    keep_df_exp = pd.get_dummies(keep_df, columns = cat_names_new)
    

    non_cat_names_in_keep_df = []

    for name in list(keep_df.columns.values):
        if name not in cat_names_new:
            non_cat_names_in_keep_df.append(name)
    

    # removing all rows where rent is not applicable or given

    keep_df_exp['uf17'].replace([99999],[np.NaN],inplace=True)
    keep_df_exp['uf17'].replace([7999],[np.NaN],inplace=True)
    keep_df_exp_new = keep_df_exp[keep_df_exp.uf17.notnull()]


    #keep_df_exp_new is the expanded chosen columns after dropping all non applicable rent rows

    filter_df = keep_df_exp_new

    X,y = filter_df.loc[:,filter_df.columns != 'uf17'], filter_df.loc[:,'uf17']


    X = X.as_matrix()
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)
    
    ridge_pipe = make_pipeline(MaxAbsScaler(), Ridge())
    param_grid = {'ridge__alpha': np.logspace(.1,1,10)}
    grid = GridSearchCV(ridge_pipe, param_grid, cv = 5)
    grid.fit(X_train,y_train)
    
    return grid, X_test, y_test

  


# In[19]:

def score_rent():
    
    """
    Builds the model and computes the prediction accuracy using the test data sets.
    
    Returns the R squared score for the model.
    
    Parameters
    ----------

    None
    
    Returns
    -------
        
    r2_score : float
        Returns the floating point value of the score of applying the model to the testing dataset.
      
    Notes
    -----
    R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, 
    or the coefficient of multiple determination for multiple regression.
    R-squared = Explained variation / Total variation
     
    """
    
    grid, X_test, y_test = build_model()
    r2_score = grid.score(X_test,y_test)
    return r2_score
    
        
def predict_rent():
    
    """
    Computes the predicted values for the response variable for a given model on the test data set.
    
    Returns the test data, the true values and the predicted values.

    Parameters
    ----------

    None
    
    Returns
    -------
    X_test : ndarray
        Numpy array holding the feature matrix for the test set.
    
    y_test : ndarray
        Numpy array holding values for response variable for the test set.
        
    y_pred : ndarray
        Numpy array holding the predicted values for each observation in the testing features data set.
    
    
    Notes
    -----
      
    """
    
    grid, X_test, y_test = build_model()
    print(grid)
    y_pred = grid.predict(X_test)
    return X_test,y_test,y_pred




