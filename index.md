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
```python
import pandas as pd
data = pd.read_excel('20013-present.xlsx')
data.head
data.head
```
```

Statistic Date	Distribution Zip	Children Served	Adults Served	Seniors Served	Total Served	Distribution City	Distribution State
0	2013-12-31	10001	8040	190036	12690	210766	NaN	NaN
1	NaT	10002	5771	240308	27600	273679	NaN	NaN
2	NaT	10003	2830	14977	9383	27190	NaN	NaN
3	NaT	10005	950	603	7	1560	NaN	NaN
4	NaT	10009	41179	171251	49099	261529	NaN	NaN
```


Here is a link to our website: https://nycity-meals.glitch.me/

## Interactive Website
For the Third project, we implemented machine learning to analyze how different aspects of a community impact food insecurity. Since we not only had data on New York City's meal gap, but also have access to public census data, we think that we can run a model with inputs as some chosen feature of a certain neiborhood, and output as food insecurity indicators. This would serve as a tool for understating correlation between the made up of the neiborhood and its food insecurity problem. Furthermore, it could potentially been used as a predictive model.  

The model we chose is Lasso Regression, which is commonly known for getting input-output like relationship for a Machine Learning regression. Moreover, Lasso regression allow us to tune the freedom of the model in order to find which are the most correlated features. As freedom decrease, Lasso model would assign zero weight to less correlated input features. This behavior leads us to the final 5 features that we use for the model.  

During the training process, 10 out of 195 NTA district were selected randomly as validation set. And 10% of the remaining data were chose randomly as the test data set. The model was trained until the loss (mse) on validation set is about 0.01(normalized).  

We decided to present the model with a interactive website, the website is coded using Dash, a python framworks for building data visualization web page. The deplyment of the website is done using Heroku. Now, try out wenbsite, Design Your Own City!

```python
plt.figure(figsize=(12,8))
plt.hist(twitter_df_clean['rating_numerator'], bins=np.arange(min(twitter_df_clean['rating_numerator']), twitter_df_clean.rating_numerator.quantile(.99), 1), color="teal")
plt.title('Distribution of WeRateDogs dog rating', fontsize=16)
plt.xlabel('dog rating (value out of 10)')
plt.show()
```



---

# AI Pioneer


