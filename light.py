#!/usr/bin/env python
# coding: utf-8

# In[201]:


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from matplotlib import pyplot as plt
import lightgbm as lgb

# In[116]:


pd.options.display.max_colwidth=100
np.set_printoptions(linewidth=140,edgeitems=10)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[117]:


D_train = pd.read_csv("data/train.csv")
D_train.head()


# In[118]:


D_test = pd.read_csv("data/test.csv")
D_test.head()


# In[119]:


D_test.index = range(1 + len(D_train), len(D_test) + len(D_train) + 1)
D_test.head()


# In[120]:


train_ids = D_train.index.values
test_ids = D_test.index.values


# In[121]:


D = pd.concat([D_train, D_test], sort=False)
D.head()


# In[122]:


del D_train, D_test


# ## Preprocessing

# In[123]:


def process_date(date):
    years = []
    months = []
    days = []
    weekdays = []
    hours = []
    minutes = []
    
    for d in date:
        date_obj = datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
    
        years.append(date_obj.year)
        months.append(date_obj.month)
        days.append(date_obj.day)
        weekdays.append(date_obj.isoweekday())
        hours.append(date_obj.hour)
        minutes.append(date_obj.minute)
        
    return (years, months, days, weekdays, hours, minutes)


# In[300]:


res = process_date(D["Время отправления по расписанию"])
data = D.copy()
data["year"] = res[0]
data["month"] = res[1]
data["day"] = res[2]
data["weekday"] = res[3]
data["hours"] = res[4]
data["minutes"] = res[5]
data.head()





data.drop(columns=["Дата рейса", "Время отправления по расписанию", "Время отправления фактическое", 
                   "Время прибытия по расписанию", "1 КЗ Код", "Время прибытия фактическое"], inplace=True)
data.head()


# In[254]:


# from_ap = data['А/П отправл'].values
# to_ap = data['А/П прибыт'].values
# from_to = []
# for i in range(len(data)):
#     from_to.append(from_ap[i] + "_" + to_ap[i])
# data["from_to"] = from_to
# del from_ap, to_ap, from_to


# ## One-hot feature encoding

# In[303]:


data = pd.get_dummies(data, prefix=["from"], columns=["А/П отправл"])
data.drop(columns="А/П прибыт", inplace=True)
# data.drop(columns=["А/П отправл", "А/П прибыт"], inplace=True)
data.head()


# In[304]:


data.head()


# ## Train test split

# In[306]:


X = data.drop(columns=["Задержка отправления в минутах"])
Y = data["Задержка отправления в минутах"]


# In[307]:


x_train, x_test, y_train, y_test = train_test_split(X.loc[train_ids], Y.loc[train_ids], 
                                                    test_size=0.1, random_state=5)


# ## Models

# In[308]:


def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 6, 
    'learning_rate': 0.12,
    'verbose': 1, 
    'num_leaves' : 12,
    'early_stopping_round': 30}
n_estimators = 85


d_train = lgb.Dataset(x_train, label=y_train)
d_test = lgb.Dataset(x_test, label=y_test)
watchlist = [d_test]

model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=1)

preds = model.predict(x_test)
err = rmse(y_test, preds)
print('RMSLE = ' + str(err))


result = model.predict(X.loc[test_ids])


# In[324]:


Result = pd.DataFrame(result, index=range(0, len(Y.loc[test_ids])), columns=["Задержка отправления в минутах"])
Result.head()


# In[325]:


res = list(map(lambda x: x if x > 0 else 0.0, Result["Задержка отправления в минутах"].values))
Result["Задержка отправления в минутах"] = res


# In[326]:


Result.to_csv("lgb.csv", index_label="index")


# In[ ]:




