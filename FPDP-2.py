# Databricks notebook source
# MAGIC %md
# MAGIC # Image hashing problem
# MAGIC Image hashing is the process of using an algorithm to assign a unique hash value (digital fingerprint) to an image.
# MAGIC Use cases:
# MAGIC - Near-Duplicate Image Detection
# MAGIC - Content-based Image Retrieval
# MAGIC - Copyright Protection
# MAGIC - Digital Forensics
# MAGIC - so on...
# MAGIC
# MAGIC Theoretically, hash algorithms are simple, but they involve intensive computation. Think Bitcoin mining.
# MAGIC
# MAGIC Imagine we have an incrementally loaded dataset comprising images uploaded by users, and during the ingestion, we need to give a digital fingerprint for each image? Or we might need to detect duplicate images within a database containing tens of thousands of pictures.

# COMMAND ----------

#%pip install opencv-python
%pip install pathos
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
#import cv2
import numpy as np
import io
import os
import hashlib
import base64

def generate_random_base64(n, length):
    def generate_random_base64_string():
        random_binary_data = os.urandom(length)
        base64_string = base64.b64encode(random_binary_data).decode('utf-8')
        return base64_string
    
    random_strings = [generate_random_base64_string() for _ in range(length)]
    data = {'img': random_strings}
    df = pd.DataFrame(data)
    return df

def hash_img(x):
    return hashlib.sha256(x.encode('utf-8')).hexdigest()

# COMMAND ----------

df = generate_random_base64(5000, 1000)
ori_df = df.copy(deep = True)

# COMMAND ----------

df.shape

# COMMAND ----------

def hash_loop(df):
    hashes = []
    for _, row in df.iterrows():
        hashes.append(hash_img(row['img']))
    df['hash'] = hashes
    return df

hash_loop(df)

# COMMAND ----------

def hash_apply(df):
    df['hash'] = df['img'].apply(hash_img)
    return df

hash_apply(df)

# COMMAND ----------

from pathos.multiprocessing import ProcessingPool as Pool

def hash_multi_processors(df):
    with Pool() as pool:  # Use pathos ProcessingPool
        hashes = pool.map(hash_img, df['img'].values)
    df['hash'] = hashes
    return df

# COMMAND ----------

hash_multi_processors(df)

# COMMAND ----------

import concurrent.futures

def hash_multi_threads(df):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        hashes = list(executor.map(hash_img, df['img'].values))
    df['hash'] = hashes
    return df

# COMMAND ----------

hash_multi_threads(df)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType, StringType
import pyspark.pandas as ps

# COMMAND ----------

psdf = ps.from_pandas(ori_df)

# COMMAND ----------

display(psdf)

# COMMAND ----------

psdf['img'].apply(hash_img)

# COMMAND ----------

# MAGIC %md
# MAGIC Best practice: define a udf and boradcast to the workers across cluster 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Takeaway
# MAGIC - Do a lot of experiments
# MAGIC - Carefully choose data structure
# MAGIC - Loop is the last resort, vectoriza if you can
# MAGIC - Think horizontal scaling, parallelism, look at spark, databricks, it's also for distrubuted computation

# COMMAND ----------


