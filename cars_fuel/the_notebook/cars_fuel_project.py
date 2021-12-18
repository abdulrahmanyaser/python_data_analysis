#!/usr/bin/env python
# coding: utf-8

# # Cars Fuel 

# # Description of Data:
# 
# **these are the columns that the data set contain :**
# 
# 
# **Model** – vehicle make and model
# 
# **Displ** – engine displacement in liters
# 
# **Cyl** – number of engine cylinders
# 
# **Trans** – transmission type plus number of gears :
# 
#         1 - Auto - Automatic
# 
#         2 - Man - Manual
# 
#         3 - SemiAuto - Semi-Automatic
# 
#         4 - SCV - Selectable Continuously Variable (e.g. CVT with paddles)
# 
#         5 - AutoMan - Automated Manual
# 
#         6 - AMS - Automated Manual-Selectable (e.g. Automated Manual with paddles)
# 
#         7 - Other - Other
# 
#         8 - CVT - Continuously Variable
# 
#         9 - CM3 - Creeper/Manual 3-Speed
# 
#         10 - CM4 - Creeper/Manual 4-Speed
# 
#         11 - C4 - Creeper/Manual 4-Speed
# 
#         12 - C5 - Creeper/Manual 5-Speed
# 
#         13 - Auto-S2 - Semi-Automatic 2-Speed
# 
#         14 - Auto-S3 - Semi-Automatic 3-Speed
# 
#         15 - Auto-S4 - Semi-Automatic 4-Speed
# 
#         16 - Auto-S5 - Semi-Automatic 5-Speed
# 
#         17 -Auto-S6 - Semi-Automatic 6-Speed
# 
#         18 - Auto-S7 - Semi-Automatic 7-Speed
# 
# **Drive** – 2-wheel Drive, 4-wheel drive/all-wheel drive
# 
# **Fuel** – fuel(s)
# 
# ##### Cert Region :**
#         1- CA - California
#         2- CE - Calif. + NLEV (Northeast trading area)
#         3- CF - Clean Fuel Vehicle
#         4- CL - Calif. + NLEV (All states)
#         5- FA - Federal All Altitude
#         6- FC - Tier 2 Federal and Calif.
#         7- NF - CFV + NLEV(ASTR) + Calif.
#         8- NL - NLEV (All states)
#         
# **Stnd** – vehicle emissions standard code.
# 
# **Stnd Description** – vehicle emissions standard description. **--> for info :**  https://www.epa.gov/greenvehicles/federal-and-california-light-duty-vehicle-emissions-standards-airpollutants
# 
# 
# **Underhood ID** – engine family or test group ID. **--> for info :**
# http://www.fueleconomy.gov/feg/findacarhelp.shtml#airPollutionScore
# 
# **Veh Class** – EPA vehicle class. **--> for info :** 
# http://www.fueleconomy.gov/feg/findacarhelp.shtml#epaSizeClass
# 
# **Air Pollution Score (Smog Rating).**  **--> for info** http://www.fueleconomy.gov/feg/findacarhelp.shtml#airPollutionScore and https://www.epa.gov/greenvehicles/smog-rating
# 
# **City MPG** – city fuel economy in miles per gallon
# 
# **Hwy MPG** – highway fuel economy in miles per gallon
# 
# **Cmb MPG** – combined city/highway fuel economy in miles per gallon
# 
# **Greenhouse Gas Score (Greenhouse Gas Rating)**  **--> for info :** https://www.epa.gov/greenvehicles/greenhouse-gas-rating
# 
# **SmartWay** – Yes, No, or Elite. **--> for info :**  https://www.epa.gov/greenvehicles/consider-smartwayvehicle
# 
# **Comb CO2** – combined city/highway CO2 tailpipe emissions in grams per mile
# 
# 
# ## Table of Contents
# <ul>
# <li><a href="#investigating  dataset">investigating the datasets</a></li>
# <li><a href="#wrangling">wrangling data</a></li>
# <li><a href="#the visuals">Exploring with Visuals</a></li>    
# </ul>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
np.set_printoptions(suppress = True,linewidth=100,precision=2)


# <a id='investigating  dataset'></a>
# ## investigating the datasets:
#     number of samples in each dataset
#     number of columns in each dataset
#     duplicate rows in each dataset
#     datatypes of columns
#     features with missing values
#     number of non-null unique values for features in each dataset
#     what those unique values are and counts for each

# In[2]:


df08 = pd.read_csv("C:/Users/mantabanta/Desktop/fuel_economy/fuel_economy_datasets/all_alpha_08.csv",delimiter=",")
df18 = pd.read_csv("C:/Users/mantabanta/Desktop/fuel_economy/fuel_economy_datasets/all_alpha_18.csv",delimiter=";")


# In[3]:


# at first will start with 2008 data set
df08.head()


# In[4]:


df08.describe()


# In[5]:


df08.shape


# we 've 2404 rows(samples) and 18 columns(features)

# In[6]:


df08.duplicated().sum()


# In[7]:


df08.info()


# In[8]:


df08.isna().sum()


# ##### there is a lot of missing data here 

# In[9]:


df08.nunique()


# In[10]:


df08.nunique


# In[ ]:





# In[11]:


# now for 2018 dataset
df18.head()


# In[12]:


df18.describe()


# In[13]:


df18.shape


# ##### we 've 1611 rows(samples) and 18 columns(features)

# In[14]:


df18.duplicated().sum()


# In[15]:


df18.info()


# In[16]:


df18.isna().sum()


# ##### the info shows that 2 columns have missing data (Displ , Cyl)

# In[17]:


df08.nunique()


# In[18]:


df18.nunique


# In[ ]:





# <a id='wrangling'></a>
# # wrangling data    
# <ul>
# <li><a href="#Drop Extraneous Columns">dropping non useful features</a></li>
# <li><a href="#Rename features">renaming features</a></li>
# <li><a href="#missing data">missing data</a></li>
# <li><a href="#Drop duplicated">duplicated data</a></li>
# <li><a href="#Fixing data types">Fixing Data Types</a></li>
# <li><a href="#Filter by Certification Region">filtering data</a></li>
# </ul>

# <a id='Drop Extraneous Columns'></a>
# ## Drop Extraneous Columns

# By looking at the description of the features, there are some features that are of no use to us
# 
# So let's define them and remove 'em from both datasets
# 
# #___> From 2008 dataset: 'Stnd', 'Underhood ID', 'FE Calc Appr', 'Unadj Cmb MPG'
# 
# #___> From 2018 dataset: 'Stnd', 'Stnd Description', 'Underhood ID', 'Comb CO2'

# In[19]:


# if u might ask why i don't use a heat map to show the relations and choose the weakest features ,
# here is why
# heat map for 2008
Correlation_Matrix = df08.corr()
f, ax = plt.subplots(figsize=(11, 11))
sns.heatmap(Correlation_Matrix,annot=True)
plt.title('Correlation among features');
plt.show()


# In[20]:


# heat map for 2018
# for 2008
Correlation_Matrix = df18.corr()
f, ax = plt.subplots(figsize=(11, 11))
sns.heatmap(Correlation_Matrix,annot=True)
plt.title('Correlation among features');
plt.show()


# as shown above the columns needs a lot of work so the corr() won't be useful here so a manual search 'd be better .
# i coul've removed these features after the wrangling process but i like cars so 'll just do it manually (an execuse for reading xd xd xd )
# 

# In[21]:


# drop columns from 2008 dataset
df08.drop(['Stnd', 'Underhood ID', 'FE Calc Appr', 'Unadj Cmb MPG'], axis=1, inplace=True)

# confirm changes
df08.columns


# In[22]:


# drop columns from 2018 dataset
df18.drop(['Stnd', 'Stnd Description', 'Underhood ID', 'Comb CO2'], axis=1, inplace=True)

# confirm changes
df18.columns


# In[ ]:





# <a id='Rename features'></a>
# # Rename features

# In[23]:


# Change the "Sales Area" column label in the 2008 dataset only to "Cert Region" for consistency , 
# since it's "Cert Region" in 2018 dataset .

# Rename all column labels to replace spaces with underscores and convert everything to lowercase.


# In[24]:


# rename Sales Area to Cert Region
df08.rename({"Sales Area":"cert region"},axis=1,inplace=True)

# confirm changes
df08.head(1)


# In[25]:


# replace spaces with underscores and lowercase labels for 2008 dataset
df08.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)

# confirm changes
df08.columns


# In[26]:


# replace spaces with underscores and lowercase labels for 2018 dataset
df18.rename(columns=lambda x:x.strip().lower().replace(" ","_"),inplace=True)

# confirm changes
df18.columns


# In[27]:


# make sure they're all identical like this
(df08.columns == df18.columns).all()


# In[ ]:





# <a id='missing data'></a>
# # missing data

# In[28]:


df08.isna().sum()


# In[29]:


# let's take a look at some columns with missing data for Curiosity . let's say cmb_mpg &  cyl
df08[df08.cmb_mpg.isna()]


# In[30]:


# now for cyl column
df08[df08.cyl.isna()]


# In[31]:


# let's drop the missing data for 2008 dataset
df08.dropna(inplace=True)


# In[32]:


# now  for 2018 dataset cyl & displ
df18.isna().sum()


# In[33]:


# let's look at 'em
df18[df18.cyl.isna()]


# In[34]:


# let's check if maybe we can fill the missing data for 2018 dataset 
# will check all the data for the KIA Soul Electric model and see the results 
df18.query("model == 'KIA Soul Electric'")


# #### sadly the KIA  model certified only once in FA & CA and both are the ones that contains missing data :(

# In[35]:


# now let's drop the missing data for 2018 too
df18.dropna(inplace=True)


# In[ ]:





# <a id='Drop duplicated'></a>
# # Drop duplicated

# In[36]:


# let's check for both ifles
df08.duplicated().sum()


# In[37]:


df18.duplicated().sum()


# In[38]:


# let's look at both file duplicated data

df08.loc[df08.duplicated(keep=False),:] 


# In[39]:


df18.loc[df18.duplicated(keep=False),:] 


# In[40]:


# not let's drop 'em in both files
df08.drop_duplicates(inplace=True)
df18.drop_duplicates(inplace=True)


# In[41]:


# print number of duplicates again to confirm dedupe - should both be 0
df08.duplicated().sum(), df18.duplicated().sum()


# In[42]:


df08.shape , df18.shape


# In[ ]:





# <a id='Fixing data types'></a>
# # Fixing data types

# In[43]:


df08.dtypes


# In[44]:


df08.head()


# In[45]:


# let's work on 'em one by one
# check value counts for the 2008 cyl column
df08['cyl'].value_counts()


# In[46]:


# Extract int from strings in the 2008 cyl column
df08['cyl'] = df08['cyl'].str.extract("(\d+)").astype(np.int64)


# In[47]:


# Check value counts for 2008 cyl column again to confirm the change
df08['cyl'].value_counts()


# In[48]:


# checking for 2018 file
df18.dtypes


# In[49]:


# as we see all the values are suppose to be int not float 
df18['cyl'].value_counts()


# In[50]:


# convert 2018 cyl column to int
df18['cyl'] = df18['cyl'].astype(np.int64)


# In[51]:


df08.dtypes


#  we have some problems with these columns [fuel,air_pollution_score ,city_mpg ,hwy_mpg ,cmb_mpg ,greenhouse_gas_score] .
#  
#  all of them must be integers but an error raises when converting them due to some of the data cells have some division process in 'em , as shown in hwy_mpg column in the next cell .
#  
#  but  According to this [this link](http://www.fueleconomy.gov/feg/findacarhelp.shtml#airPollutionScore) , it says :  
#  "If a vehicle can operate on more than one type of fuel, an estimate is provided for each fuel type."
#  
#  so this is not a problem  of miss typed data or some division  it's a special case ,
#  obviously this problem in both files :(

# In[52]:


# First, let's get all the hybrids in 2008
hy08 = df08[df08['fuel'].str.contains('/')]
hy08


# In[53]:


df_before=hy08.copy() # df_before refers to the data that on the left side of the / 
df_after=hy08.copy()  # df_after refers to the data that on the right side of the / 


# In[54]:


split_columns = ['fuel', 'air_pollution_score', 'city_mpg', 'hwy_mpg', 'cmb_mpg', 'greenhouse_gas_score']

# apply split function to each column of each dataframe copy
for c in split_columns:
    df_before[c] = df_before[c].apply(lambda x: x.split("/")[0])
    df_after[c] = df_after[c].apply(lambda x: x.split("/")[1])


# In[55]:


df_after


# In[56]:


# combine dataframes to add to the original dataframe
new_rows = df_before.append(df_after)

# now we have separate rows for each fuel type of each vehicle!
new_rows


# In[57]:


# drop the original hybrid rows
df08.drop(hy08.index, inplace=True)

# add in our newly separated rows
df08 = df08.append(new_rows, ignore_index=True)


# In[58]:


# check that all the original hybrid rows with "/"s are gone
df08[df08['fuel'].str.contains('/')]


# In[59]:


# now for the 2018 file 
hy18 = df18[df18.fuel.str.contains("/")]
hy18.head(1)


# this time only [fuel, city_mpg, hwy_mpg, cmb_mpg] need some work the air_pollution_score & greenhouse_gas_score already integers thx god :) , but there is another problem beside the / one . some of these columns [ city_mpg, hwy_mpg, cmb_mpg]
# data has the first part of the 2 types of fuel like this ??????- as shown in the next cell 

# In[60]:


hy18.city_mpg.value_counts()


# In[61]:


hy18.hwy_mpg.value_counts()


# In[62]:


hy18.cmb_mpg.value_counts()


# In[63]:


mpg_columns = ['city_mpg', 'hwy_mpg', 'cmb_mpg']

# apply split function to each column of each dataframe copy
for c in mpg_columns:
    df18[c] = df18[c].apply(lambda x: x.replace("??????-",''))


# In[64]:


# let's check if it worked
df18[df18.city_mpg.str.contains("-")]


# In[65]:


#nice now we can solve the / problem 
hy18 = df18[df18.fuel.str.contains("/")]
hy18


# In[66]:


# but first let's remove the cells that don't contain / , i really don't know why are they included :( in these columns [city_mpg	hwy_mpg	cmb_mpg]
# like the example down
hy18.city_mpg.value_counts()


# In[67]:


hy18.hwy_mpg.value_counts()


# In[68]:


hy18.cmb_mpg.value_counts()


# In[69]:


# removing them by the index of the values 
city_remove = hy18.query("city_mpg == ['16','17','18','13']").index
city_remove


# In[70]:


hy18.drop(index=city_remove,axis=1,inplace=True)


# In[71]:


# let's check
hy18.city_mpg.value_counts()


# In[72]:


hy18.hwy_mpg.value_counts()


# In[73]:


hy18.cmb_mpg.value_counts()


# In[ ]:





# In[74]:


# good now i can work on the data safely 
df_18before=hy18.copy() # doing the same as the df_before
df_18after=hy18.copy()  # doing the same as the df_after


#  this time only [fuel, city_mpg, hwy_mpg, cmb_mpg] need some work the air_pollution_score & greenhouse_gas_score already integers thx god :)

# In[75]:


split_columns = ['fuel' , 'city_mpg' , 'hwy_mpg' , 'cmb_mpg']

# apply split function to each column of each dataframe copy
for c in split_columns:
    df_18before[c] = df_18before[c].apply(lambda x: x.split("/")[0])
    df_18after[c] = df_18after[c].apply(lambda x: x.split("/")[1])


# In[76]:


# checking if it worked for all the 4 columns
df_18before[df_18before['fuel'].str.contains('/')]


# In[ ]:





# In[77]:


df_18before[df_18before['city_mpg'].str.contains('/')]


# In[ ]:





# In[78]:


df_18before[df_18before['hwy_mpg'].str.contains('/')]


# In[ ]:





# In[79]:


df_18before[df_18before['cmb_mpg'].str.contains('/')]


# In[ ]:





# In[80]:


# combine dataframes to add to the original dataframe
new_rows = df_18before.append(df_18after)

# now we have separate rows for each fuel type of each vehicle!
new_rows


# In[81]:


# let's create a set of the indexies that we wanna remove from the original df18

indexies = set()

for x in df18[df18['fuel'].str.contains('/')].index :
    indexies.add(x)

for x in df18[df18['hwy_mpg'].str.contains('/')].index :
    indexies.add(x)

for x in df18[df18['cmb_mpg'].str.contains('/')].index :
    indexies.add(x)


# In[82]:


# drop the original hybrid rows
df18.drop(indexies, inplace=True)
# add in our newly separated rows
df18 = df18.append(new_rows, ignore_index=True)


# In[83]:


# now let's check if it worked , will try fuel column
df18[df18['fuel'].str.contains('/')]


# In[ ]:





# In[84]:


df08.shape , df18.shape


# In[85]:


# now let's continue the data type adjusting :)
# convert mpg columns to floats
mpg_columns = ['city_mpg', 'hwy_mpg', 'cmb_mpg']
for c in mpg_columns:
    df18[c] = df18[c].astype(float)
    df08[c] = df08[c].astype(float)


# In[86]:


# fixing air_pollution_score column b  converting the data type from float to int
df08['air_pollution_score'] = df08['air_pollution_score'].astype(float)


# In[87]:


# fixing greenhouse_gas_score column b  converting the data type from float to int
df08['greenhouse_gas_score'] = df08['greenhouse_gas_score'].astype(np.int64)


# #### All the dataypes are now fixed! Take one last check to confirm all the changes

# In[88]:


# checking the data types of both files
df08.dtypes == df18.dtypes


# #### the air_pollution_score column is different in both files because in the df08 there are some float data but in df18  all of the data are integers 

# <a id='Filter by Certification Region'></a>
# # Filter by Certification Region

# In[89]:


df08.cert_region.value_counts()


# In[90]:


df18.cert_region.value_counts()


# #### since FA & CA are in both data sets so w'll filter based one of 'em to answer the questions cuz am tired of this :( 

# In[91]:


# filter datasets for rows following California standards
ca_08 = df08.query("cert_region == 'CA'").copy()
ca_18 = df18.query("cert_region == 'CA'").copy()


# In[92]:


ca_08.cert_region.value_counts() , ca_18.cert_region.value_counts()


# In[93]:


# drop certification region columns form both datasets since all the data we have already related to CA certification
ca_08.drop('cert_region', axis=1, inplace=True)
ca_18.drop('cert_region', axis=1, inplace=True)


# In[94]:


ca_08.shape , ca_18.shape


# In[95]:


ca_08.columns


# In[96]:


ca_18.columns


# In[ ]:





# <a id='the visuals'></a>
# # the visuals 
# 
#     Compare the distributions of greenhouse gas score in 2008 and 2018 .
# 
#     How has the distribution of combined mpg changed from 2008 to 2018 ?
# 
#     Describe the correlation between displacement and combined mpg .
# 
#     Describe the correlation between greenhouse gas score and combined mpg .
# 
#     How strong are the features relations in both 2008 and 2018 files ?

# In[ ]:





# ####  distributions of greenhouse gas score
# 

# In[97]:


n, bins, patches = plt.hist(x=[df08['greenhouse_gas_score'] , df18['greenhouse_gas_score']] , 
                            label=['greenhouse gas score 2008','greenhouse gas score 2018'],
                            bins=10, color = ['#31736e','#060003'],alpha=0.7, rwidth=0.85)
plt.legend()
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('greenhouse gas score 2008 & 2018 distribution');


# In[ ]:





# #### combined mpg distribution change

# In[98]:


n, bins, patches = plt.hist(x=[df08['cmb_mpg'] , df18['cmb_mpg'] ] , bins=9, 
                            color=['#ffcb12' , '#133337' ], label=['2008_cmb' , '2018_cmb'] ,alpha=0.7, rwidth=0.85)
plt.legend()
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('combined mpg 2008 & 2018 distribution ');


# In[ ]:





# #### displacement and combined mpg correlation

# In[99]:


# for 2008 
df08.plot(x='cmb_mpg'  ,y= 'displ' , kind='scatter', color = '#03c03c' );


# In[100]:


# for 2018
df18.plot(x='cmb_mpg'  ,y= 'displ' , kind='scatter', color = '#ff4040' );


# In[ ]:





# #### greenhouse gas score and combined mpg correlation

# In[101]:


# for 2008 
df08.plot(x='cmb_mpg'  ,y= 'greenhouse_gas_score' , kind='scatter' , color = '#7f2020');


# In[102]:


# for 2018
df18.plot(x='cmb_mpg'  ,y= 'greenhouse_gas_score' , kind='scatter' , color = '#11215b');


# In[ ]:





# #### the features relations

# In[103]:


# for 2008
Correlation_Matrix = df08.corr()
f, ax = plt.subplots(figsize=(11, 11))
sns.heatmap(Correlation_Matrix,annot=True)
plt.title('Correlation among features');
plt.show()


# In[104]:


# for 2018
Correlation_Matrix = df18.corr()
f, ax = plt.subplots(figsize=(11, 11))
sns.heatmap(Correlation_Matrix,annot=True)
plt.title('Correlation among features');
plt.show()


# In[ ]:





# In[107]:


# Saving the final CLEAN datasets as new files!
df08.to_csv('clean_fuel_08.csv', index=False)
df18.to_csv('clean_fuel_18.csv', index=False)


# In[ ]:




