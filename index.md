# NYC Food Inecurity
---
## Overview
City Harvest is New York City’s largest food rescue organization. Currently, helping to feed more than 1.2 million New Yorkers struggling to find food, City Harvest has the goal to end hunger in communities by distributing food, educating the community, and finding solutions to end food insecurity.

Our goal in working with City Harvest is to provide an overview of the Food Landscape in neighborhoods, aiming to identify group needs and explore solutions on a community level. Through this process we will be completing three different projects to gain further insight into understanding what factors affect the meals served in New York City.

## Exploratory Poster
For the first project, we could only acess to a data containing all the CityHarvest affiliated food agency and their address at that time. So with limited data to work with, we decided to run a exploratory data analysis and made a poster illustrating some information.  
Through our analysis, we picked up 3 NTA district and compare them on the number of food banks, meal served, and neiborhood's average income. The poster is shown below.
<img src="pic/poster.png?raw=true"/>

## Time Leap Map
For the second project we created a website that charted out the meals served in the various city community districts over the years. We implemented this visual so that the user can click through the various years and see how the distribution and concentration of meals served changed over the years in New York City. 
We cleaned up over 15 years of data from CityHarvest and FeedNYC on New York's overall meal served. Then density map is vreated using Tableau, and the website is developed in glitch.com framwork.  
  
A few entry of the data set looks like the following, for confidential reason, we could not upload the full dataset.  

```python
import pandas as pd
data = pd.read_excel('2013-present.xlsx')
data.head()
```

<img src="pic/data.png?raw=true"/>

Now, lets preceed to out time leap website!

[<img src="pic/mapweb.png?raw=true"/>](https://cityharvest.glitch.me)


Here is a link to our website: [Click Here](https://nycity-meals.glitch.me/)

## Interactive Website
For the Third project, we implemented machine learning to analyze how different aspects of a community impact food insecurity. Since we not only had data on New York City's meal gap, but also have access to public census data, we think that we can run a model with inputs as some chosen feature of a certain neiborhood, and output as food insecurity indicators. This would serve as a tool for understating correlation between the made up of the neiborhood and its food insecurity problem. Furthermore, it could potentially been used as a predictive model.  
  
For the made up of a neiborhood, we decided to look into 5 features of any area, they are

1. Houeshold poverty
2. Unemployment Rate
3. Population with College Degree
4. Population with Highschool Degree
5. Population with Citizenship

We cleaned up NYC's NTA level census data and combined some of the columns in the way we want to derive those parameters. A few entry of the data set is shown below:  
 <img src="pic/data_nta.png?raw=true"/> 
  
The model we chose is Lasso Regression, which is commonly known for getting input-output like relationship for a Machine Learning regression. Moreover, Lasso regression allow us to tune the freedom of the model in order to find which are the most correlated features. As freedom decrease, Lasso model would assign zero weight to less correlated input features. This behavior leads us to the final 5 features that we use for the model.  
  
We use Scikitlearn package for our regression, here is the code for the training
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('data_model_3.csv', delimiter=',', index_col=0)
df = df.drop(['Unnamed: 1'], axis=1)
X = df[['Total population', '  Male', '  Female', 'per', 'un']]
Y = df['served']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
Y_test = Y_test.values
Y_train = Y_train.values

clf = svm.SVR(kernel='rbf', C=1e9, gamma=0.1)
clf.fit(X_train, Y_train)
result = clf.predict(X_test)
accuracy_score(Y_te, result)
```
During the training process, 10 out of 195 NTA district were selected randomly as validation set. And 10% of the remaining data were chose randomly as the test data set. The model was trained until the loss (mse) on validation set is about 0.01(normalized).  

We decided to present the model with a interactive website, the website is coded using Dash, a python framworks for building data visualization web page. The deplyment of the website is done using Heroku. Now, try our wenbsite, [Design Your Own City!](http://cityharvest-app.herokuapp.com/)

[<img src="pic/app.png?raw=true"/>](http://cityharvest-app.herokuapp.com/)


## Code and Dataset
All relative code is uploaded on to my gthub [repository](https://github.com/raymondminglee/CityHarvest-DataVisualization), Due to confidential reason, we could not share our dataset online. 


---

# AI Pioneer


