import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import MinMaxScaler


#LOAD THE DATA
df = pd.read_csv("MOCK_DATA (1).csv")

#CHECK ANY NA VALUE

na_check = df.isna().sum()

#DATA IS SHOWN IN NOMINAL OR ORDINAL DATA,IT SHOULD BE CHANGED BY ONE HOT ENCODING
#WE ALSO GONNA DROP LAST COLUMN TO AVOID DUMMY VARIABEL TRAP

dummies_gender = pd.get_dummies(df.gender)
dummies_gender_final = dummies_gender.drop(columns="M")

#SAME AS BEFORE USING ONE HOT ENCODING TO CHANGE INCOME

dummies_income = pd.get_dummies(df.income)
dummies_income_final = dummies_income.drop(columns="High")

#SAME AS BEFORE USING ONE HOT ENCODING TO CHANGE INTEREST
dummies_interest = pd.get_dummies(df.interest)
dummies_interest_final = dummies_interest .drop(columns="War")

#SAME AS BEFORE USING ONE HOT ENCODING TO CHANGE CITY

dummies_city = pd.get_dummies(df.city)
dummies_city_final = dummies_city.drop(columns="aceh")


#COMBINE ALL VARIABEL TO A SINGLE DATAFRAME

merged = pd.concat([df,dummies_gender_final,dummies_income_final,dummies_city_final,dummies_interest_final],axis="columns")


#THERE ARE STILL A FEW STRING OR CATEGORICAL DATA AND NEED TO DROP

df_final = merged.drop(columns=["id","first_name","last_name","email","gender","interest","income","city"])

#DATA IS TO HIGH


ss = StandardScaler()
ss.fit(df_final)
scaled_data = ss.transform(df_final)

#MODELLING USING K MEANS CLUSTERING
km = KMeans(n_clusters=5)
y_predicted = km.fit_predict(scaled_data)

#SHOWING ALL DATA WITH CLUSTER
merged["cluster"] = y_predicted


#VISUALIZATION DATA

"""THE FIRST VISUALIZATION WE WANT TO SEEING BETWEEN INTEREST AND SALARY"""

new_merged = merged.copy(deep=True) # COPY DATAFRAME

#THE DATA IS TOO BIG SO TO MINIMIZE WE ARE USING MIN MAX SCALER

scaler = MinMaxScaler()
minmax = scaler.fit_transform(new_merged[["salary"]])
new_merged["new_salary"] = minmax

#IN THIS CASE WE CANNOT USE ONE HOT ENCODING CAUSE WILL MAKE TOO MANY COLUMN SO WE ARE USING LABEL ENCODER

label = LabelEncoder()
encod = label.fit_transform(new_merged["interest"])

new_merged["new_interest"] = encod

df0 = new_merged[new_merged.cluster ==0]
df1 = new_merged[new_merged.cluster ==1]
df2 = new_merged[new_merged.cluster ==2]
df3 = new_merged[new_merged.cluster ==3]
df4 = new_merged[new_merged.cluster ==4]

plt.scatter(df0["new_interest"],df0.salary,color='black')
plt.scatter(df1["new_interest"],df1.salary,color='red')
plt.scatter(df2["new_interest"],df2.salary,color='blue')
plt.scatter(df3["new_interest"],df3.salary,color='orange')
plt.scatter(df4["new_interest"],df4.salary,color='green')

plt.xlabel("Interest")
plt.ylabel("Salary")
plt.show()

"""THE RESULT IS NOT SHOWING ANY SIGN OF CLUSTER,THE DATA IS TOO CLOSE EACH OTHER"""



"""SECOND ANALYST WE ARE USING PCA TO CHANGE THE DATA POINT
HERE ARE 2 OPTION TO PLOT,YOU CAN USE STATIC PLOT BY USING MATPLOTLIB AND DYNAMIC BY USING PLOTLY"""

reduced_data = PCA(n_components=2).fit_transform(scaled_data)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
centroids = km.cluster_centers_

#PLOTLY RENDERED IN BROWSER HTML

fig = px.scatter(results,x="pca1", y="pca2",template="plotly_dark",title="Scatter Plot PCA")
pio.renderers.default = "browser"
fig.show()

#MATPLOTLIB

plt.scatter(results["pca1"],y=results["pca2"])
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.xlabel("pca1")
plt.ylabel("pca2")
plt.show()

"""IN PCA WE ARE SEE TWO BIG CLUSTER BUT THE CENTROID IS NOT EQUALLY PRODUCED SO OUR DATA
IS NOT SHOWING ANY INSIGHT"""
