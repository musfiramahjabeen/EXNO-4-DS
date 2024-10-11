# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```python
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/7ec14dd8-7a19-46ed-8ef5-111d32cd0ea8)
```python
df.dropna()
```
![image](https://github.com/user-attachments/assets/7a643efd-e783-4e6f-8bcd-6500ae9f631d)
```python
# TYPE CODE TO FIND MAXIMUM VALUE FROM HEIGHT AND WEIGHT FEATURE

max_height = df['Height'].max()
max_weight = df['Weight'].max()

print("Maximum Height:",max_height)
print("Maximum Weight:",max_weight)
```
![image](https://github.com/user-attachments/assets/19caaaae-c6f2-4912-866e-3832afdd3b7b)
```python
#Perform minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/ea1da995-4900-42d9-a9d4-dc1ac731b922)
```python
#Perform standard scaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/062c2843-f0ca-406a-8f96-5f2a5dbf0e2e)
```python
#Perform Normalizer

from sklearn.preprocessing import Normalizer

scaler = Normalizer()
df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/7dad27ec-7682-4eb8-966b-b46a9d302d67)
```python
#Perform MaxAbsScaler

from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/49642099-8b87-4153-8ea5-8130258343d6)
```python
#Perform RobustScaler

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/1aadcf31-8a86-4caf-94a1-4002ff867058)

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```

```python
data = pd.read_csv('income.csv',na_values=["?"])
data
```


![image](https://github.com/user-attachments/assets/0d1b73dc-6ca3-4564-80e6-3756f69b5685)



```python
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/7d8ce515-feae-4e3e-b198-9d3cd14d6af5)


```python
m = data[data.isnull().any(axis=1)]
m
```

![image](https://github.com/user-attachments/assets/51be0af9-fb19-4fa1-9a64-36768c090642)

```python
data2=data.dropna(axis=0)
data2
```


![image](https://github.com/user-attachments/assets/a07a9b96-7310-43e0-85a5-f9b14f0c10dc)

```python
sal = data['SalStat']
data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![image](https://github.com/user-attachments/assets/b7a9bc35-7371-47a0-9310-2bbfb6e60b93)

```python
sal2=data2['SalStat']
sal = data['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/user-attachments/assets/8644b554-0fb8-4c9b-b52d-93f3ccdc5e0c)


```python
data2
```

![image](https://github.com/user-attachments/assets/88df1e07-14a2-4e3e-b23c-47cd05124c55)


```python
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![image](https://github.com/user-attachments/assets/f05fc2bd-ff2f-4b6e-a67c-78abba27de79)

```python
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/ef58fee0-5e1d-4341-b9fc-c14d691f8063)


```python
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/user-attachments/assets/e53ab964-690d-40e0-8da7-613ee3683b8e)


```python
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/user-attachments/assets/41e7c6dd-05ec-4756-9a48-07d1bc74b9a4)

```python
features=list(set(columns_list)-set(['SalStat']))
x=new_data[features].values
print(x)
```

![image](https://github.com/user-attachments/assets/cc104885-55c7-49bf-9ed0-c0973cb5e8e6)

```python
# splitting the data int train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
#fitting the values for x and y
KNN_classifier.fit(train_x,train_y)
```

![image](https://github.com/user-attachments/assets/01aea97f-1b1e-4233-b595-99f7d362b5a6)

```python
# predicting the test values with values
prediction = KNN_classifier.predict(test_x)
# performance metric ckeck
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```

![image](https://github.com/user-attachments/assets/21b02fc7-04bf-4525-8fbc-ed5bf5e73996)

```python
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```

![image](https://github.com/user-attachments/assets/c410a188-3cfa-4314-8944-9040639dcebc)


```python
print('Misclassified samples: %d' % (test_y != prediction).sum())
```

![image](https://github.com/user-attachments/assets/9daf6906-6b5a-477d-86c9-c23c142f5096)

```python
data.shape
```

![image](https://github.com/user-attachments/assets/3dca7cdf-dbdb-4d64-8ebf-211a9f96e375)

# RESULT:
 The Feature scaling and feature selection executed successfully for the given data.
