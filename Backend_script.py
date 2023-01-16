import pyspark
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from pathlib import Path
import pickle
import pandas as pd
from google.cloud import storage

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')

# if the following command generates an error, you probably didn't enable 
# the cluster security option "Allow API access to all Google Cloud services"
# under Manage Security → Project Access when setting up the cluster
!gcloud dataproc clusters list --region us-central1

!pip install -q google-cloud-storage==1.43.0
!pip install -q graphframes

# if nothing prints here you forgot to include the initialization script when starting the cluster
!ls -l /usr/lib/spark/jars/graph*

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.sql import SQLContext
from graphframes import *

spark

# Put your bucket name below and make sure you can access it without an error
bucket_name = 'elad_318640828' 
full_path = f"gs://{bucket_name}/"
paths=[]

client = storage.Client()
blobs = client.list_blobs(bucket_name)
for b in blobs:
    if b.name != 'graphframes.sh':
        paths.append(full_path+b.name)
        
parquetFile = spark.read.parquet(*paths)
doc_text_pairs = parquetFile.select("text", "id").rdd
doc_title_pairs = parquetFile.select("title", "id").rdd
doc_anchor_pairs = parquetFile.select("anchor_text", "id").rdd

# Count number of wiki pages
parquetFile.count()

# if nothing prints here you forgot to upload the file inverted_index_gcp.py to the home dir
%cd -q /home/dataproc
!ls inverted_index_gcp.py

# adding our python module to the cluster
sc.addFile("/home/dataproc/inverted_index_gcp.py")
sys.path.insert(0,SparkFiles.getRootDirectory())

from inverted_index_gcp import InvertedIndex

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124


def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS


# PLACE YOUR CODE HERE
def word_count(text, id):
    ''' Count the frequency of each word in `text` (tf) that is not included in
    `all_stopwords` and return entries that will go into our posting lists.
    Parameters:
    -----------
      text: str
        Text of one document
      id: int
        Document id
    Returns:
    --------
      List of tuples
        A list of (token, (doc_id, tf)) pairs
        for example: [("Anarchism", (12, 5)), ...]
    '''
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]

    final_tokens = []
    for tok in tokens:
        if tok not in all_stopwords:
            final_tokens.append(tok)
    tf_dict = Counter(final_tokens)
    returning = []
    for elem in tf_dict.items():
        returning.append((elem[0], (id, elem[1])))
    return returning


##################
def reduce_word_counts(unsorted_pl):
    ''' Returns a sorted posting list by wiki_id.
    Parameters:
    -----------
      unsorted_pl: list of tuples
        A list of (wiki_id, tf) tuples
    Returns:
    --------
      list of tuples
        A sorted posting list.
    '''
    finale = []
    finale = sorted(unsorted_pl, reverse=False)
    return finale


##################
def calculate_df(postings):
    ''' Takes a posting list RDD and calculate the df for each token.
    Parameters:
    -----------
      postings: RDD
        An RDD where each element is a (token, posting_list) pair.
    Returns:
    --------
      RDD
        An RDD where each element is a (token, df) pair.
    '''
    finale = []
    finale = postings.map(lambda x: (x[0], len(x[1])))

    return finale


##################
def partition_postings_and_write(postings):
    ''' A function that partitions the posting lists into buckets, writes out
    all posting lists in a bucket to disk, and returns the posting locations for
    each bucket. Partitioning should be done through the use of `token2bucket`
    above. Writing to disk should use the function  `write_a_posting_list`, a
    static method implemented in inverted_index_colab.py under the InvertedIndex
    class.
    Parameters:
    -----------
      postings: RDD
        An RDD where each item is a (w, posting_list) pair.
    Returns:
    --------
      RDD
        An RDD where each item is a posting locations dictionary for a bucket. The
        posting locations maintain a list for each word of file locations and
        offsets its posting list was written to. See `write_a_posting_list` for
        more details.
    '''

    file_location_dictionary = {}
    lama = postings.map(lambda x: (token2bucket_id(x[0]), x)).groupByKey().map(
        lambda x: InvertedIndex.write_a_posting_list(x, bucket_name))

    return lama

# time the index creation time
t_start = time()
# word counts map
word_counts_text = doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))
word_counts_title = doc_title_pairs.flatMap(lambda x: word_count(x[0], x[1]))
word_counts_anchor = doc_anchor_pairs.flatMap(lambda x: word_count(x[0], x[1]))

postings_text = word_counts_text.groupByKey().mapValues(reduce_word_counts)
postings_title = word_counts_title.groupByKey().mapValues(reduce_word_counts)
postings_anchor = word_counts_anchor.groupByKey().mapValues(reduce_word_counts)
# filtering postings and calculate df
postings_filtered_text = postings_text.filter(lambda x: len(x[1])>50)
postings_filtered_title = postings_title.filter(lambda x: len(x[1])>50)
postings_filtered_anchor = postings_anchor.filter(lambda x: len(x[1])>50)

w2df_text = calculate_df(postings_filtered_text)
w2df_title = calculate_df(postings_filtered_title)
w2df_anchor = calculate_df(postings_filtered_anchor)

w2df_dict_text = w2df_text.collectAsMap()
w2df_dict_title = w2df_title.collectAsMap()
w2df_dict_anchor = w2df_anchor.collectAsMap()

# partition posting lists and write out
_ = partition_postings_and_write(postings_filtered_text).collect()
_ = partition_postings_and_write(postings_filtered_title).collect()
_ = partition_postings_and_write(postings_filtered_anchor).collect()

index_const_time = time() - t_start

# test index construction time
print(index_const_time)

# collect all posting lists locations into one super-set
super_posting_locs_body = defaultdict(list)
for blob in client.list_blobs(bucket_name, prefix='postings_gcp_body'):
  if not blob.name.endswith("pickle"):
    continue
  with blob.open("rb") as f:
    posting_locs = pickle.load(f)
    for k, v in posting_locs.items():
      super_posting_locs_body[k].extend(v)

# collect all posting lists locations into one super-set
super_posting_locs_title = defaultdict(list)
for blob in client.list_blobs(bucket_name, prefix='postings_gcp_title'):
  if not blob.name.endswith("pickle"):
    continue
  with blob.open("rb") as f:
    posting_locs = pickle.load(f)
    for k, v in posting_locs.items():
      super_posting_locs_title[k].extend(v)

# collect all posting lists locations into one super-set
super_posting_locs_anchor = defaultdict(list)
for blob in client.list_blobs(bucket_name, prefix='postings_gcp_anchor'):
  if not blob.name.endswith("pickle"):
    continue
  with blob.open("rb") as f:
    posting_locs = pickle.load(f)
    for k, v in posting_locs.items():
      super_posting_locs_anchor[k].extend(v)

# Create inverted index instance
inverted = InvertedIndex()
# Adding the posting locations dictionary to the inverted index
inverted.posting_locs = super_posting_locs_body
# Add the token - df dictionary to the inverted index
inverted.df = w2df_dict_text
# write the global stats out
inverted.write_index('.', 'index_body')
# upload to gs
index_src = "index_body.pkl"
index_dst = f'gs://{bucket_name}/postings_gcp_body/{index_src}'
!gsutil cp $index_src $index_dst

!gsutil ls -lh $index_dst

# Create inverted index instance
inverted = InvertedIndex()
# Adding the posting locations dictionary to the inverted index
inverted.posting_locs = super_posting_locs_title
# Add the token - df dictionary to the inverted index
inverted.df = w2df_dict_text
# write the global stats out
inverted.write_index('.', 'index_title')
# upload to gs
index_src = "index_title.pkl"
index_dst = f'gs://{bucket_name}/postings_gcp_title/{index_src}'
!gsutil cp $index_src $index_dst

!gsutil ls -lh $index_dst

# Create inverted index instance
inverted = InvertedIndex()
# Adding the posting locations dictionary to the inverted index
inverted.posting_locs = super_posting_locs_anchor
# Add the token - df dictionary to the inverted index
inverted.df = w2df_dict_text
# write the global stats out
inverted.write_index('.', 'index_anchor')
# upload to gs
index_src = "index_anchor.pkl"
index_dst = f'gs://{bucket_name}/postings_gcp_anchor/{index_src}'
!gsutil cp $index_src $index_dst

!gsutil ls -lh $index_dst