import gzip
import numpy as np
import struct
from azureml.core import Workspace, Dataset
import pandas as pd 
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from azureml.core import Workspace, Dataset



def load_model(path):
  subscription_id = '<subscription_id>'
  resource_group = '<resource_group>'
  workspace_name = '<workspace_name>'

  workspace = Workspace(subscription_id, resource_group, workspace_name)

  df = Dataset.get_by_name(workspace, name=path)
  df = df.to_pandas_dataframe()

  return df

# flag = 0 for train data, flag = 1 for test data
def feature_engineering(X, flag = 0, training_cols = None):
  X = X[['first_contact_date','origin','seller_id']]

  #Convert date to year, month, day, quarter
  X['year'] = pd.DatetimeIndex(X['first_contact_date']).year
  X['month'] = pd.DatetimeIndex(X['first_contact_date']).month
  X['day'] = pd.DatetimeIndex(X['first_contact_date']).day
  X['quarter'] = X['month'].apply(lambda x:x//4)

  #Drop contact date and seller id
  X.drop(columns=['first_contact_date','seller_id'], axis=1, inplace=True)

  X = pd.get_dummies(X, drop_first=True, prefix='', prefix_sep='')

  if flag == 1:
    X = X.T.reindex(training_cols).T.fillna(0)

  return X, X.columns


def prepare_train_validation_data(closed_deals_path, market_lead_path):

  closed_deals = load_model(closed_deals_path)
  market_lead = load_model(market_lead_path)
  
  #Join the two datasets
  mf = pd.merge(market_lead, closed_deals, left_on='mql_id', right_on='mql_id', how='left')

  # Create target variable
  mf['converted'] = mf[['seller_id']].where(mf[['seller_id']].isnull()==True, 1).fillna(0).astype(int)

  X = mf.loc[:, mf.columns != 'converted']
  y = mf.loc[:, ['converted']]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

  X_train, training_cols = feature_engineering(X_train)

  X_test, cols = feature_engineering(X_test, 1, training_cols)

  return X_train, X_test, y_train, y_test, training_cols

def prepare_test_data(closed_deals_path, market_lead_path, training_cols):

  closed_deals = load_model(closed_deals_path)
  market_lead = load_model(market_lead_path)
  
  #Join the two datasets
  mf = pd.merge(market_lead, closed_deals, left_on='mql_id', right_on='mql_id', how='left')

  # Create target variable
  mf['converted'] = mf[['seller_id']].where(mf[['seller_id']].isnull()==True, 1).fillna(0).astype(int)

  X = mf.loc[:, mf.columns != 'converted']
  y = mf.loc[:, ['converted']]


  X, cols = feature_engineering(X_train, training_cols)

  return X


# To evaluate model performance
def evaluate(model, X_test):
  y_pred = model.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  acc_score = accuracy_score(y_test, y_pred)

  return y_pred, cm, acc_score
