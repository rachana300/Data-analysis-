#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd;
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("car_price_dataset.csv")
df


# In[2]:


df.info()


# In[3]:


df.head(5)


# In[4]:


df.isnull()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.duplicated().sum()


# In[8]:


plt.figure(figsize=(8, 5))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


# In[9]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Mileage'], y=df['Price'])
plt.title("Price vs. Mileage")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.show()


# In[10]:


plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Price'])
plt.title("Boxplot of Car Prices")
plt.xlabel("Price")
plt.show()


# In[ ]:


sns.pairplot(df, diag_kind="kde")
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
sns.countplot(y=df['Brand'], order=df['Brand'].value_counts().index)
plt.title("Count of Cars by Brand")
plt.xlabel("Count")
plt.ylabel("Brand")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 5))
sns.violinplot(x=df['Fuel_Type'], y=df['Price'])
plt.title("Price Distribution by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Price")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 5))
df.groupby('Transmission')['Price'].mean().plot(kind='bar', color=['skyblue', 'orange'])
plt.title("Average Price by Transmission Type")
plt.xlabel("Transmission")
plt.ylabel("Average Price")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 5))
df.groupby('Year')['Price'].mean().plot(marker='o', linestyle='-', color='g')
plt.title("Trend of Car Prices Over the Years")
plt.xlabel("Year")
plt.ylabel("Average Price")
plt.grid(True)
plt.show()


# In[ ]:


plt.figure(figsize=(7, 7))
df['Fuel_Type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title("Fuel Type Distribution")
plt.ylabel("")  # Hide y-label for better visualization
plt.show()


# In[ ]:


plt.figure(figsize=(7, 7))
df['Fuel_Type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title("Fuel Type Distribution")
plt.ylabel("")  # Hide y-label for better visualization
plt.show()


# In[ ]:


plt.figure(figsize=(8, 5))
for fuel in df['Fuel_Type'].unique():
    sns.kdeplot(df[df['Fuel_Type'] == fuel]['Price'], label=fuel)
plt.title("Kernel Density Estimate of Price by Fuel Type")
plt.xlabel("Price")
plt.legend()
plt.show()


# In[ ]:


df['Car_Age'] = 2025 - df['Year']  # Assuming current year is 2025
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Car_Age'], y=df['Price'])
plt.title("Price Distribution by Car Age")
plt.xlabel("Car Age")
plt.ylabel("Price")
plt.show()


# In[ ]:


Model=pd.get_dummies(df['Model'])
print(Model)


# In[ ]:


Fuel_Type=pd.get_dummies(df['Fuel_Type'])
print(Fuel_Type)


# In[ ]:


Transmission=pd.get_dummies(df['Transmission'])
print(Transmission)


# In[ ]:


Brand=pd.get_dummies(df['Brand'])
print(Brand)


# In[ ]:


df.Brand.value_counts()


# In[ ]:




