import random
import requests

from flask import Flask, request, jsonify

from collections import Counter, OrderedDict
import pandas as pd
import re
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from pathlib import Path
from google.cloud import storage, client
import math
from contextlib import closing
import inverted_index_gcp
from inverted_index_gcp import MultiFileReader, InvertedIndex
import pickle
import itertools
from itertools import islice, count, groupby
import os
from pathlib import Path
from time import time
from timeit import timeit
import struct

nltk.download('stopwords')

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


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


def read_posting_list(inverted, w, drive):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, drive)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


# Uploading the page rank
def upload_page_rank(drivePR):
    pagerank = pd.read_csv(drivePR + 'part-00000-2b72797c-f127-41a5-bd89-658334f84c5c-c000.csv.gz', compression='gzip',
                           header=0, sep=' ', quotechar='"', error_bad_lines=False)
    pagerank.columns = ["id"]
    new = pagerank["id"].str.split(",", n=1, expand=True)
    pagerank["id"] = new[0]
    pagerank["page rank"] = new[1]
    pagerank.set_index('id')
    dict_scores = pagerank['page rank'].to_dict()
    return dict_scores


# Uploading the page view
def upload_page_view(drivePV):
    with open(drivePV + "wid2pv.pkl", 'rb') as f:
        page_View = dict(pickle.loads(f.read()))
    return page_View

# Uploading the id2titles
def upload_page_to_titles(drive_to_title):
    with open(drive_to_title + "id2titles1.pkl", 'rb') as f:
        page_to_title = dict(pickle.loads(f.read()))
    return page_to_title
# path_to_body_II = "./posting_gcp/"
#
# project_id = 'myfirstgcp-370210'
# data_bucket_name = 'elad_318640828'
#
# with open(f'./postings_gcp/index_body.pkl', 'rb') as f:
#     index_body = pickle.load(f)

drivePR = './pr/pr/'
drivePV = './pageview/'
driveBody = "./postings_gcp_body/postings_gcp/"
driveTitle = "./postings_gcp_title/postings_gcp/"
driveAnchor = "./postings_gcp_anchor/postings_gcp/"
index_body = InvertedIndex.read_index(driveBody, "index_body")
index_title = InvertedIndex.read_index(driveTitle, "index_title")
index_anchor = InvertedIndex.read_index(driveAnchor, "index_anchor")
page_View = upload_page_view(drivePV=drivePV)
page_Rank = upload_page_rank(drivePR=drivePR)
id2titles = upload_page_to_titles(drive_to_title=driveBody)

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

def check_condition(l_r_b, l_r_t, l_r_a):
    # some conditions
    if l_r_b <= 0.2:
        l_r_b = 0.2
    elif l_r_b >= 1:
        l_r_b = 1
    if l_r_t <= 0.05:
        l_r_t = 0.05
    elif l_r_t >= 1:
        l_r_t = 1
    if l_r_a <= 0.05:
        l_r_a = 0.05
    elif l_r_a >= 1:
        l_r_a = 1
    return [l_r_b, l_r_t, l_r_a]

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # # BEGIN SOLUTION
    res_total = []

    search_body_results = search_body(query, True)
    search_title_results = search_title(query, True)
    search_anchor_text_results = search_anchor(query, True)

    # Combine results into a single dataframe
    res_total.append([search_body_results, 0])
    res_total.append([search_title_results, 0])
    res_total.append([search_anchor_text_results, 0])

    results = []
    # counting the length of the lists
    l_b = len(search_body_results)
    l_t = len(search_title_results)
    l_a_t = len(search_anchor_text_results)

    # For comparison in the future with the query
    with_stop_words_query_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]

    query_tokens = []
    for tok in with_stop_words_query_tokens:
        if tok not in all_stopwords:
            query_tokens.append(tok)
    query_tokens = Counter(query_tokens)
    length_of_filtered_query = len(query_tokens)

    # Assign different weights to each method's output
    if '?' in query:
        if len(query) > length_of_filtered_query + 2:
            learning_rate_body = 0.2
            learning_rate_title = 0.3
            learning_rate_anchor = 0.50
            minus_chosen = 0.02
            plus_others = 0.01
        else:
            learning_rate_body = 0.3
            learning_rate_title = 0.35
            learning_rate_anchor = 0.35
            minus_chosen = 0.02
            plus_others = 0.01
        for _ in range(40):
            proc_list = [1, 2, 3]
            weights = (learning_rate_body, learning_rate_title, learning_rate_anchor)
            print(weights)
            what_to_take = random.choices(proc_list, weights=weights)
            print(what_to_take)
            if what_to_take[0] == 1:
                doc_title = res_total[0][0]
                if res_total[0][1] >= l_b:
                    continue
                results.append(doc_title[res_total[0][1]])
                res_total[0][1] += 1
                learning_rate_title += plus_others
                learning_rate_anchor += plus_others
                learning_rate_body -= minus_chosen
                # some conditions
                lst_learn = check_condition(learning_rate_body, learning_rate_title, learning_rate_anchor)
                learning_rate_body = lst_learn[0]
                learning_rate_title = lst_learn[1]
                learning_rate_anchor = lst_learn[2]

            if what_to_take[0] == 2:
                doc_title = res_total[1][0]
                if res_total[0][1] >= l_t:
                    continue
                results.append(doc_title[res_total[1][1]])
                res_total[1][1] += 1
                learning_rate_body += plus_others
                learning_rate_anchor += plus_others
                learning_rate_title -= minus_chosen
                # some conditions
                lst_learn = check_condition(learning_rate_body, learning_rate_title, learning_rate_anchor)
                learning_rate_body = lst_learn[0]
                learning_rate_title = lst_learn[1]
                learning_rate_anchor = lst_learn[2]
            if what_to_take[0] == 3:
                doc_title = res_total[2][0]
                if res_total[0][1] >= l_a_t:
                    continue
                results.append(doc_title[res_total[2][1]])
                # print(doc_title[res_total[0][1]])
                res_total[2][1] += 1
                learning_rate_body += plus_others
                learning_rate_title += plus_others
                learning_rate_anchor -= minus_chosen
                lst_learn = check_condition(learning_rate_body, learning_rate_title, learning_rate_anchor)
                learning_rate_body = lst_learn[0]
                learning_rate_title = lst_learn[1]
                learning_rate_anchor = lst_learn[2]
        # return results
    else:
        if len(query) > length_of_filtered_query + 2:
            learning_rate_body = 0.40
            learning_rate_title = 0.05
            learning_rate_anchor = 0.55
            minus_chosen = 0.05
            plus_others = 0.025
        else:
            learning_rate_body = 0.2
            learning_rate_title = 0.5
            learning_rate_anchor = 0.3
            minus_chosen = 0.05
            plus_others = 0.025

        for _ in range(40):
            proc_list = [1, 2, 3]
            weights = (learning_rate_body, learning_rate_title, learning_rate_anchor)
            print(weights)
            what_to_take = random.choices(proc_list, weights=weights)
            print(what_to_take)
            if what_to_take[0] == 1:
                doc_title = res_total[0][0]
                if res_total[0][1] >= l_b:
                    continue
                results.append(doc_title[res_total[0][1]])
                res_total[0][1] += 1
                learning_rate_title += plus_others
                learning_rate_anchor += plus_others
                learning_rate_body -= minus_chosen
                # some conditions
                lst_learn = check_condition(learning_rate_body, learning_rate_title, learning_rate_anchor)
                learning_rate_body = lst_learn[0]
                learning_rate_title = lst_learn[1]
                learning_rate_anchor = lst_learn[2]

            if what_to_take[0] == 2:
                doc_title = res_total[1][0]
                if res_total[1][1] >= l_t:
                    continue
                results.append(doc_title[res_total[1][1]])
                res_total[1][1] += 1
                learning_rate_body += plus_others
                learning_rate_anchor += plus_others
                learning_rate_title -= minus_chosen
                # some conditions
                lst_learn = check_condition(learning_rate_body, learning_rate_title, learning_rate_anchor)
                learning_rate_body = lst_learn[0]
                learning_rate_title = lst_learn[1]
                learning_rate_anchor = lst_learn[2]
            if what_to_take[0] == 3:
                doc_title = res_total[2][0]
                if res_total[2][1] >= l_a_t:
                    continue
                results.append(doc_title[res_total[2][1]])
                # print(doc_title[res_total[0][1]])
                res_total[2][1] += 1
                learning_rate_body += plus_others
                learning_rate_title += plus_others
                learning_rate_anchor -= minus_chosen
                lst_learn = check_condition(learning_rate_body, learning_rate_title, learning_rate_anchor)
                learning_rate_body = lst_learn[0]
                learning_rate_title = lst_learn[1]
                learning_rate_anchor = lst_learn[2]
        # return results



    results = [item[0] for item in groupby(results)]
    print(results)
    # results = results.sort_values(by='weighted_score', ascending=False)
    # print(results)
    # # END SOLUTION
    # '''
    # wiki_id = []
    # for tup in results:
    #     wiki_id.append(tup[0])
    # all_the_page_ranks = get_pagerank(wiki_ids=wiki_id)
    # rank_dict = {}
    # iteration = 0
    # while iteration < len(wiki_id):
    #     rank_dict[wiki_id[iteration]] = all_the_page_ranks[iteration]
    #     iteration += 1
    # result_sorted_by_page_rank = sorted(results, key=lambda x: rank_dict[x[0]], reverse=True)
    # return jsonify(result_sorted_by_page_rank)
    # '''
    # print(results)
    return jsonify(results)


@app.route("/search_body")
def search_body(query=None, from_search=False):
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    with_stop_words_query_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]

    query_tokens = []
    for tok in with_stop_words_query_tokens:
        if tok not in all_stopwords:
            query_tokens.append(tok)
    query_tokens = Counter(query_tokens)

    # this is normaliztion of the query for the cosine similarity
    norm_factor = 0
    for term, freq in query_tokens.items():
        norm_factor += freq ** 2
    norm_factor = math.sqrt(norm_factor)
    if norm_factor == 0:
        return jsonify([])

    sim_dict = {}
    norm_factor_doc = 0
    for term, freq in query_tokens.items():
        posting_list_per_term = read_posting_list(index_body, term, driveBody)
        for doc_id, score in posting_list_per_term:
            norm_factor_doc = 0
            if doc_id in sim_dict:
                sim_dict[doc_id] += score
            else:
                sim_dict[doc_id] = score
            norm_factor_doc += score ** 2
    norm_factor_docs_final = math.sqrt(norm_factor_doc)

    similarity_final_dict = dict(sorted(sim_dict.items(), key=lambda x: (x[1] / (norm_factor * norm_factor_docs_final)), reverse=True))

    res = list()
    iteration = 0
    for doc_id in similarity_final_dict.keys():
        if iteration < 100:
            res.append((doc_id, id2titles[doc_id]))
            iteration += 1
        else:
            break
    # END SOLUTION
    if from_search:
        return res
    return jsonify(res)

@app.route("/search_title")
def search_title(query=None, from_search=False):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    dic = {}
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    for term in tokens:
        if index_anchor.df.get(term):
            ls_doc_freq = read_posting_list(index_title, term, driveTitle)
            for doc, freq in ls_doc_freq:
                dic[doc] = dic.get(doc, 0) + freq
    lst_doc = Counter(dic).most_common()
    for item in lst_doc:
        if item[0] not in id2titles:
            continue
        res.append((item[0], id2titles[item[0]]))
    # END SOLUTION
    if from_search:
        return res
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor(query=None, from_search=False):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    dic = {}
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    for term in tokens:
        if index_anchor.df.get(term):
            ls_doc_freq = read_posting_list(index_anchor, term, driveAnchor)
            for doc, freq in ls_doc_freq:
                dic[doc] = dic.get(doc, 0) + freq
    lst_doc = Counter(dic).most_common()
    # print(lst_doc)
    iter = 0
    for item in lst_doc:
        # print(iter)
        if item[0] not in id2titles:
            continue
        res.append((item[0], id2titles[item[0]]))
        # iter += 1
    # END SOLUTION
    if from_search:
        return res
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank(wiki_ids=None):
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        try:
            res.append(page_Rank[doc_id])
        except:
            res.append(0)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview(wiki_ids=None):
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # print(page_View)
    for doc_id in wiki_ids:
        try:
            res.append(page_View[doc_id])
        except:
            res.append(0)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
