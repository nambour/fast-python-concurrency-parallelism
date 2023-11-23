# Databricks notebook source
# MAGIC %md
# MAGIC # Fast Python Data Processing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem
# MAGIC Firstly, let's consider a simple numerical computation problem. In data science, our initial step involves making essential transformations to the training data to meet the requirements of the models, or we call feature engineering. This includes processes such as normalization and standardization.
# MAGIC
# MAGIC In Python, we commonly perform min-max normalization to scale data that they end up ranging between 0 and 1. The algorithm is indeed straightforward.
# MAGIC
# MAGIC X' = (X - min(X)) / (max(X) - min(X))

# COMMAND ----------

# MAGIC %md
# MAGIC Now we set up a few helper functions
# MAGIC - Generate a dataframe contains some random values
# MAGIC - Calculate the new scaled value 

# COMMAND ----------

import random
import numpy as np
import pandas as pd

def gen_random_df(n_rows):
    data = {'c1': [random.randint(0, 1000) for _ in range(n_rows)],
            'c2': [random.randint(0, 1000) for _ in range(n_rows)],
            'c3': [random.randint(0, 1000) for _ in range(n_rows)]}
    return pd.DataFrame(data)

def normalize(val, min_val, max_val):
    if min_val == max_val:
        return 0
    else:
        return (val - min_val)/(max_val - min_val)

df = gen_random_df(100000)

# COMMAND ----------

df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loop
# MAGIC Now we have a 1,000 rows dataframe. Let's try to scale the column c1.
# MAGIC
# MAGIC Your intuition might be leaning towards the idea that we only require a looping structure to iterate through each row in c1, calculating the values row by row. This approach should be efficient enough to handle the simple computation swiftly.

# COMMAND ----------

def normalize_iterrows(df):
    t = []
    for _, row in df.iterrows():
        t.append(normalize(row['c1'], df['c1'].min(), df['c1'].max()))
    df['norm_c1'] = t
    return df

normalize_iterrows(df)

# COMMAND ----------

# MAGIC %md
# MAGIC But what if for 1 million rows? In ML world it's not uncommon for dataset with millions rows and thoundsands features.

# COMMAND ----------

# MAGIC %md
# MAGIC 30s x 10 x 100 = 30000s = 8+ hours for a dataset has 1 million rows and 100 features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply optimized loop
# MAGIC Suddenly, you might recall that Pandas offers an apply function capable of transforming a set of data. That's a great idea! Of course, at its core, it likely involves a looping structure. However, I believe that it's already optimizad using Cython. (Complied Pyhton offers comparable performace to C)

# COMMAND ----------

def normalize_apply(df):
    df['norm_c1'] = df['c1'].apply(normalize, args = (df['c1'].min(), df['c1'].max()))
    return df

normalize_apply(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vectorization
# MAGIC The previous results might be good enough, indeed. Perhaps, if it's already close to 5 PM, you can relax and start winding down, contemplating what to eat for dinner. 
# MAGIC
# MAGIC However, as you lean back in your chair, a term you recently stumbled upon in a tech blog, "Vectorization," suddenly springs to mind. But what exactly is Vectorization? You promptly return to your computer to delve into research.
# MAGIC
# MAGIC In mathematics, it is the linear transformation which converts the matrix into a vector. Linear algebra, numerical analysis is my worst subjects while I was in the college. So I won't talk more about the algorithm itself. But here more relevantly, in Pandas, vectorization means the way pandas implement it which has batch API to process multiple items of data at once and native loop wihout calling back Python, this gives you lightning speed for numerical computation.

# COMMAND ----------

def normalize_vecterize(df):
    min_val = df['c1'].min()
    max_val = df['c1'].max()
    df['norm_c1'] = (df['c1'] - min_val) / (max_val - min_val)
    return df

normalize_vecterize(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Are you sure this looks like Excel formular? Yes. That's it. By runing an operation across a whole Series, Index, or even DataFrame at once, you activate the real power of pandas, leaving good engough apply fuction in the dust.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Third-party packages
# MAGIC It seems like we've arrived at an ultimate solution, right? Hold on, you should remember that many machine learning toolkits come with numerous built-in data transformation algorithms, such as scaling. In practice, we do't reinvent the wheel like this; we simply use the existing functions. Shouldn't we take a look at how these tools perform?

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

def normalize_sk(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = df['c1'].values.reshape(-1, 1)
    df['norm_c1'] = scaler.fit_transform(data)
    return df

# COMMAND ----------

normalize_sk(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Nice job scikit-learn. So you know it is pretty reliable if there's built-in function available for your job.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bonus: What about NumPy?
# MAGIC If you're familiar enough with Pandas, you'd know that Pandas is built on top of the data structures of NumPy while leveraging many of NumPy's numerical computation algorithms. So, what if we set aside Pandas and directly utilize NumPy? Could it potentially be even faster? Let's take a quick look.

# COMMAND ----------

def normalize_vecterize_np(df):
    array = df['c1'].values #convert to numpy array
    df['norm_c1'] = (array - np.min(array)) / (np.max(array) - np.min(array))
    return df

normalize_vecterize_np(df)

# COMMAND ----------

# MAGIC %md
# MAGIC No need to talk more.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC ### 100000 caculations:
# MAGIC - loop: 35s
# MAGIC - apply: 0.13s
# MAGIC - pandas vectorization: 0.07s
# MAGIC - numpy vectorization: 0.04s
# MAGIC - scikit-learn: 1.78s

# COMMAND ----------

# MAGIC %md
# MAGIC Now we move on to another problem: unstructured data

# COMMAND ----------


