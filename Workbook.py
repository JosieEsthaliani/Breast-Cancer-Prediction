#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 20:23:03 2020

@author: User
"""

import numpy as np
import pandas as pd
from scipy import stats

wdbc = pd.read_csv('wdbc.csv')
wdbc.info()

#Removing outliers wdbc
z_scores = stats.zscore(wdbc[['mean_radius', 'mean_texture','mean_perimeter'
                           ,'mean_area','mean_smoothness',
                           'mean_compactness', 'mean_concavity', 
                           'mean_concave_points','mean_symmetry',
                           'mean_fractal_dimension', 'se_radius',
                           'se_texture', 'se_parimeter', 'se_area', 
                           'se_smoothness', 'se_compactness', 'se_concavity',
                           'se_concave_points', 'se_symmetry', 'se_fractal_dimension',
                           'worst_radius', 'worst_texture', 'worst_perimeter',
                           'worst_area', 'worst_smoothness', 'worst_compactness',
                           'worst_concavity', 'worst_concave_points', 'worst_symmetry', 
                           'worst_fractal_dimension']])
abs_z_scores = np.abs(z_scores)
print("abs z score")
print(abs_z_scores)

filtered_entries = (abs_z_scores < 3).all(axis=1)
print("filteren entries")
print(filtered_entries)

new_wdbc = wdbc[filtered_entries]

print(new_wdbc)
new_wdbc.info()

#boxplot hasil 
new_wdbc.boxplot(column=['mean_radius'])
new_wdbc.boxplot(column=['mean_texture'])
new_wdbc.boxplot(column=['mean_perimeter'])
new_wdbc.boxplot(column=['mean_area'])
new_wdbc.boxplot(column=['mean_smoothness'])
new_wdbc.boxplot(column=['mean_compactness'])
new_wdbc.boxplot(column=['mean_concavity'])
new_wdbc.boxplot(column=['mean_concave_points'])

#membuat data csv sesudah bersih
new_wdbc.to_csv('wdbcBaru.csv')

#BARUUU
import numpy as np
import pandas as pd
from scipy import stats

new_wdbc = pd.read_csv('wdbcBaru.csv')
new_wdbc.info()

from sklearn import preprocessing
features_wdbc=new_wdbc.loc[:,"diagnosis":"worst_fractal_dimension"]
for column in features_wdbc.columns:
    if features_wdbc[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        features_wdbc[column] = le.fit_transform(features_wdbc[column])
features_np_wdbc=np.array(features_wdbc.values)


print(features_np_wdbc)
features_wdbc.info()
print(features_wdbc)

#HEATMAP 
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.size'] = 30
df_pilihan = features_wdbc[['diagnosis','mean_radius', 'mean_texture','mean_perimeter'
                           ,'mean_area','mean_smoothness',
                           'mean_compactness', 'mean_concavity', 
                           'mean_concave_points','mean_symmetry',
                           'mean_fractal_dimension', 'se_radius',
                           'se_texture', 'se_parimeter', 'se_area', 
                           'se_smoothness', 'se_compactness', 'se_concavity',
                           'se_concave_points', 'se_symmetry', 'se_fractal_dimension',
                           'worst_radius', 'worst_texture', 'worst_perimeter',
                           'worst_area', 'worst_smoothness', 'worst_compactness',
                           'worst_concavity', 'worst_concave_points', 'worst_symmetry', 
                           'worst_fractal_dimension']]
plt.figure(figsize=(50,25))
sns.heatmap(df_pilihan.corr(),annot=True, annot_kws={"size": 30}, linewidth=0.5)

#atribut 1
x = features_wdbc[['mean_radius']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 2
x = features_wdbc[['mean_texture']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 3
x = features_wdbc[['mean_perimeter']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 4
x = features_wdbc[['mean_area']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 5
x = features_wdbc[['mean_smoothness']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 6
x = features_wdbc[['mean_compactness']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 7
x = features_wdbc[['mean_concavity']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 8
x = features_wdbc[['mean_concave_points']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 9
x = features_wdbc[['mean_symmetry']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 10
x = features_wdbc[['mean_fractal_dimension']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 11
x = features_wdbc[['se_radius']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 12
x = features_wdbc[['se_texture']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 13
x = features_wdbc[['se_parimeter']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 14
x = features_wdbc[['se_area']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 15
x = features_wdbc[['se_smoothness']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 16
x = features_wdbc[['se_compactness']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 17
x = features_wdbc[['se_concavity']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 18
x = features_wdbc[['se_concave_points']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 19
x = features_wdbc[['se_symmetry']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 20
x = features_wdbc[['se_fractal_dimension']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 21
x = features_wdbc[['worst_radius']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 22
x = features_wdbc[['worst_texture']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 23
x = features_wdbc[['worst_perimeter']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 24
x = features_wdbc[['worst_area']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 25
x = features_wdbc[['worst_smoothness']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 26
x = features_wdbc[['worst_compactness']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 27
x = features_wdbc[['worst_concavity']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 28
x = features_wdbc[['worst_concavity']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 29
x = features_wdbc[['worst_concave_points']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 30
x = features_wdbc[['worst_symmetry']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#atribut 31
x = features_wdbc[['worst_fractal_dimension']].values
x.shape
y = features_wdbc[['diagnosis']].values 
y.shape

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#Train
model_regres = LinearRegression()  
model_regres.fit(X_train, y_train)

print(model_regres.intercept_)
print(model_regres.coef_)

y_pred = model_regres.predict(X_test)

df_comp = pd.DataFrame(y_test.flatten(), y_pred.flatten())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
r_2 = r2_score(y_test, y_pred)
print('R^2: ', r_2)

import matplotlib.pyplot as plt  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#klasifikasi
import numpy as np
import pandas as pd

dt_cancer = pd.read_csv('wdbcBaru.csv', delimiter = ',')

cancer_labels = dt_cancer[['diagnosis']]  # hasil: 1 kolom 

cancer_label_np = np.array(cancer_labels.values) # numpy array 

label_np= cancer_label_np.ravel()

# Import LabelEncoder
from sklearn import preprocessing

cancer_features_df=dt_cancer.loc[:,"diagnosis":"worst_fractal_dimension"]
for column in cancer_features_df.columns:
    if cancer_features_df[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        cancer_features_df[column] = le.fit_transform(cancer_features_df[column])
cancer_features_np=np.array(cancer_features_df.values)

cancer_labels_en = le.fit_transform(label_np)
print(cancer_labels_en)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = cancer_features_np
Y = cancer_labels_en

test = SelectKBest(score_func=chi2, k=3)
myfit = test.fit(X, Y)

np.set_printoptions(precision=3)
print(myfit.scores_)

features = myfit.transform(X)

print(features[0:1,:])

# model klasifikasi dg alg kNN

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, cancer_labels_en, test_size=0.3) 

from sklearn.neighbors import KNeighborsClassifier

kNN_model_cancer = KNeighborsClassifier(n_neighbors=2)


kNN_model_cancer.fit(X_train, y_train)

y_pred = kNN_model_cancer.predict(X_test)

from sklearn import metrics

print("Akurasi model klasifikasi dgn k=2:", metrics.accuracy_score(y_test, y_pred))

#PEMILIHAN FEATURES TERBAIK F1
cancer_labels = dt_cancer[['diagnosis']]  # hasil: 1 kolom 

cancer_label_np = np.array(cancer_labels.values) # numpy array 

label_np= cancer_label_np.ravel()

# Import LabelEncoder
from sklearn import preprocessing

cancer_features_df=dt_cancer[['mean_perimeter', 'mean_area', 'se_area', 'worst_radius', 'worst_perimeter', 'worst_area']]
for column in cancer_features_df.columns:
    if cancer_features_df[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        cancer_features_df[column] = le.fit_transform(cancer_features_df[column])
cancer_features_np=np.array(cancer_features_df.values)

cancer_labels_en = le.fit_transform(label_np)
print(cancer_labels_en)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = cancer_features_np
Y = cancer_labels_en

test = SelectKBest(score_func=chi2, k=6)
myfit = test.fit(X, Y)

np.set_printoptions(precision=3)
print(myfit.scores_)

features = myfit.transform(X)

print(features[0:1,:])

# model klasifikasi dg alg kNN

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, cancer_labels_en, test_size=0.3) 

from sklearn.neighbors import KNeighborsClassifier

kNN_model_cancer = KNeighborsClassifier(n_neighbors=2)


kNN_model_cancer.fit(X_train, y_train)

y_pred = kNN_model_cancer.predict(X_test)

from sklearn import metrics

print("Akurasi model klasifikasi dgn k=2:", metrics.accuracy_score(y_test, y_pred))

#PEMILIHAN FEATURES TERBAIK F2
cancer_labels = dt_cancer[['diagnosis']]  # hasil: 1 kolom 

cancer_label_np = np.array(cancer_labels.values) # numpy array 

label_np= cancer_label_np.ravel()

# Import LabelEncoder
from sklearn import preprocessing

cancer_features_df=dt_cancer[['mean_perimeter', 'mean_area', 'mean_concavity', 'mean_concave_points', 'worst_radius', 'worst_perimeter',  'worst_area', 'worst_concave_points']]
for column in cancer_features_df.columns:
    if cancer_features_df[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        cancer_features_df[column] = le.fit_transform(cancer_features_df[column])
cancer_features_np=np.array(cancer_features_df.values)

cancer_labels_en = le.fit_transform(label_np)
print(cancer_labels_en)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = cancer_features_np
Y = cancer_labels_en

test = SelectKBest(score_func=chi2, k=8)
myfit = test.fit(X, Y)

np.set_printoptions(precision=3)
print(myfit.scores_)

features = myfit.transform(X)

print(features[0:1,:])

# model klasifikasi dg alg kNN

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, cancer_labels_en, test_size=0.3) 

from sklearn.neighbors import KNeighborsClassifier

kNN_model_cancer = KNeighborsClassifier(n_neighbors=2)


kNN_model_cancer.fit(X_train, y_train)

y_pred = kNN_model_cancer.predict(X_test)

from sklearn import metrics

print("Akurasi model klasifikasi dgn k=2:", metrics.accuracy_score(y_test, y_pred))

# PICKLE PREDIKSI DATA BARU
import pickle
import numpy as np

pkl_filename = "cancer_Model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(kNN_model_cancer, file)

pkl_filename = "cancer_Model.pkl"  
with open(pkl_filename, 'rb') as file:  
    loaded_model_classification = pickle.load(file)

import pandas as pd 
df_new = pd.read_csv('Data_baru_prediksi_wdbc.csv')
X_new = df_new[['mean_perimeter','mean_area','mean_concavity','mean_concave_points','worst_radius','worst_perimeter','worst_area','worst_concave_points']]
arr_baru = np.array(X_new.values)
label_baru = arr_baru.ravel()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

#ubah string ke numerik
new_data_labels_en = le.fit_transform(label_baru)
new_ukuran = new_data_labels_en.reshape(8,5).transpose()

#Lakukan prediksi
y_pred_new = loaded_model_classification.predict(new_ukuran)
print("hasil prediksi : ")
print(y_pred_new)

#DECISION TREE
import time
waktu_mulai = time.time()
cancer_labels = dt_cancer[['diagnosis']]

cancer_label_np = np.array(cancer_labels.values)

label_np= cancer_label_np.ravel()

from sklearn import preprocessing

cancer_features_df=dt_cancer[['mean_perimeter','mean_area','mean_concavity','mean_concave_points','worst_radius','worst_perimeter','worst_area','worst_concave_points']]
for column in cancer_features_df.columns:
    if cancer_features_df[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        cancer_features_df[column] = le.fit_transform(cancer_features_df[column])
cancer_features_np=np.array(cancer_features_df.values)

cancer_labels_en = le.fit_transform(label_np)
print(cancer_labels_en)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, cancer_labels_en, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier 
DT_model_cancer = DecisionTreeClassifier()
DT_model_cancer.fit(X_train,y_train)

y_pred = DT_model_cancer.predict(X_test)
from sklearn import metrics
print("Model accuracy:",metrics.accuracy_score(y_test, y_pred))
print("waktu eksekusi: ", (time.time()) - waktu_mulai)

#gambar pohon keputusan
from sklearn.tree import export_graphviz
import pydotplus
dot_data = export_graphviz(DT_model_cancer,out_file=None,max_depth=5,feature_names=cancer_features_df.columns,filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png("Dtree_cancer_model.png")

