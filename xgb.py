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
from xgboost import XGBRegressor, plot_importance


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


D = pd.concat([D_train, D_test])
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


# In[ ]:


# def getFlightTable(dataset):
#     freqTable = {}
#     peucbl= np.unique(dataset['Рейс'])
#     for peuc in peucbl:
#         data = dataset.loc[dataset['Рейс']==peuc]

#         time = pd.DatetimeIndex(data['Время отправления по расписанию']).time
#         codes = data['1 КЗ Код'].values

#         for i in range(len(time)):
#             time[i] = time[i].strftime("%H:%M")

#         timeNames, timeCounts = np.unique(time, return_counts=True)

#         values = {}
#         for t,c in zip(timeNames, timeCounts):
#             tcodes = codes[np.where(time==t)]
#             freq = sum([type(a)==type(float()) for a in tcodes])
#             values.update({t: freq/c })

#         freqTable.update({str(peuc): values})
#     return freqTable

# def getNewFeature(dataset,freqTable):
#     newColumn = np.zeros(len(dataset))
#     flights = dataset['Рейс'].values
#     depTime = pd.DatetimeIndex(dataset['Время отправления по расписанию']).time

#     for i in range(len(depTime)):
#         depTime[i] = depTime[i].strftime("%H:%M")

#     def getCoeff(flight,time):
#         if str(flight) in freqTable:
#             f = freqTable[str(flight)]
#             if time in f:
#                 return f[time]
#             else:
#                 return 0 
#         else:
#             return 0

#     return [getCoeff(f,t) for f,t in zip(flights,depTime)]


# In[77]:


# data['delay_coef'] = getNewFeature(data, getFlightTable(data.loc[train_ids]))
# data.head()


# In[301]:


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


# xgb = XGBRegressor(n_estimators=85, learning_rate=0.12, n_jobs=-1, max_depth=5, reg_alpha=0.05) <br>
# xgb.fit(x_train[:100000], y_train[:100000]) <br>
# 40.73

# In[319]:


xgb = XGBRegressor(n_estimators=85, learning_rate=0.12, n_jobs=-1, max_depth=5, reg_alpha=0.05) 
xgb.fit(x_train[:175000], y_train[:175000])
y_pred1 = xgb.predict(x_test)
rmse(y_test, y_pred1)


# In[320]:


cols = x_train.columns
important = []
for i in range(len(cols)):
    if (xgb.feature_importances_[i] > 0):
        important.append(cols[i])
important


# In[321]:


xgb = XGBRegressor(n_estimators=100, learning_rate=0.12, n_jobs=-1, max_depth=5, reg_alpha=0.05)
xgb.fit(x_train[important], y_train, verbose=5)
y_pred1 = xgb.predict(x_test[important])
rmse(y_test, y_pred1)


# ## Submission

# In[322]:


xgb.fit(X.loc[train_ids][important], Y.loc[train_ids])


# In[323]:


result = xgb.predict(X.loc[test_ids][important])


# In[324]:


Result = pd.DataFrame(result, index=range(0, len(Y.loc[test_ids])), columns=["Задержка отправления в минутах"])
Result.head()


# In[325]:


res = list(map(lambda x: x if x > 0 else 0.0, Result["Задержка отправления в минутах"].values))
Result["Задержка отправления в минутах"] = res


# In[326]:


Result.to_csv("xgb.csv", index_label="index")


# In[ ]:
