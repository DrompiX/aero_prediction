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


pd.options.display.max_colwidth=100
np.set_printoptions(linewidth=140,edgeitems=10)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

D_train = pd.read_csv("data/train.csv")
D_train.head()

D_test = pd.read_csv("data/test.csv")
D_test.head()

D_test.index = range(1 + len(D_train), len(D_test) + len(D_train) + 1)
D_test.head()

train_ids = D_train.index.values
test_ids = D_test.index.values

D = pd.concat([D_train, D_test])
D.head()

del D_train, D_test

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

data = pd.get_dummies(data, prefix=["from"], columns=["А/П отправл"])
data.drop(columns="А/П прибыт", inplace=True)
data.head()

data.head()

X = data.drop(columns=["Задержка отправления в минутах"])
Y = data["Задержка отправления в минутах"]

x_train, x_test, y_train, y_test = train_test_split(X.loc[train_ids], Y.loc[train_ids], 
                                                    test_size=0.1, random_state=5)


def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


xgb = XGBRegressor(n_estimators=85, learning_rate=0.12, n_jobs=-1, max_depth=5, reg_alpha=0.05) 
xgb.fit(x_train[:175000], y_train[:175000])
y_pred1 = xgb.predict(x_test)
rmse(y_test, y_pred1)

cols = x_train.columns
important = []
for i in range(len(cols)):
    if (xgb.feature_importances_[i] > 0):
        important.append(cols[i])
important

# xgb = XGBRegressor(n_estimators=100, learning_rate=0.12, n_jobs=-1, max_depth=5, reg_alpha=0.05)
# xgb.fit(x_train[important], y_train, verbose=5)
# y_pred1 = xgb.predict(x_test[important])
# rmse(y_test, y_pred1)

xgb.fit(X.loc[train_ids][important], Y.loc[train_ids])
result = xgb.predict(X.loc[test_ids][important])

Result = pd.DataFrame(result, index=range(0, len(Y.loc[test_ids])), columns=["Задержка отправления в минутах"])
Result.head()


res = list(map(lambda x: x if x > 0 else 0.0, Result["Задержка отправления в минутах"].values))
Result["Задержка отправления в минутах"] = res

Result.to_csv("xgb.csv", index_label="index")
