from flask import Flask, request, jsonify
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
# print('importing nltk is in progress')
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import math
nltk.download('stopwords')
from inverted_index_gcp import *
# print('all import success')
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# print('dl on run.............')
DL_text = '/home/ramman/DL_text.pkl'
with open(DL_text, 'rb') as f:
    DL_text = pickle.load(f)
# print('title  on run.............')
title_id  = '/home/ramman/title_id.pkl'
with open(title_id, 'rb') as f:
    title_id = pickle.load(f)
# print('page_rank  on run.............')
page_rank  = '/home/ramman/pagerank/pagerank.pkl'
with open(page_rank, 'rb') as f:
    page_rank = pickle.load(f)
# print('page_views  on run.............')
page_views  = '/home/ramman/pageviews/pageviews-202108-user.pkl'
with open(page_views, 'rb') as f:
    page_views = pickle.load(f)

# print('title_index  on run.............')
file_name_title,title = ('/home/ramman/title_pos/','title_index')
# /home/ramman/title_pos/title_index.pkl
file_name_body,body = ('/home/ramman/body_pos/', 'body_index')

file_name_anchor,anchor = ('/home/ramman/ANCOR_pos/','anchor_index')
#find / -name title_index.* 2>/dev/null
title_index = InvertedIndex.read_index( file_name_title,  title  )
# print('body_index  on run.............')
body_index = InvertedIndex.read_index(  file_name_body,    body   )
# print('anchor_index  on run.............')
anchor_index = InvertedIndex.read_index(file_name_anchor,anchor )
#add DL to the body index
body_index.DL={}
body_index.DL = DL_text

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
english_stopwords = frozenset(stopwords.words('english'))

corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)


def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
from contextlib import closing

def read_posting_list(inverted, w, file_name):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    locs = [(file_name + lo[0], lo[1]) for lo in locs]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list

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
    # BEGIN SOLUTION
    res=Search_BM25_with_condicion(query)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
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
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    file_name = file_name_body
    res = search_body(query, body_index, file_name)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():

    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

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
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    file_name = file_name_title

    res = search_title(query, title_index, file_name)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

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
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    file_name = file_name_anchor

    res = search_anchor(query, anchor_index, file_name)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
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
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for wiki_id in wiki_ids:
        value = 0
        try:
            value = page_rank[wiki_id]
        except Exception:
            pass
        res.append(value)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
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
    res = []
    for wiki_id in wiki_ids:
        value = 0
        try:
            value = page_views[wiki_id]
        except Exception:
            pass
        res.append(value)
    # END SOLUTION
    return jsonify(res)
def search_title(query, index_title, file_name):

    query_to = tokenize(query)
    query_counter = Counter(query_to)
    title_dic = {}
    doc_lists = []

    for token in query_counter.keys():
        try:

            posting = read_posting_list(index_title, token, file_name)

            for doc_id, frequency in posting:
                if doc_id not in title_dic:
                    title_dic[doc_id] = 0
                title_dic[doc_id] += 1
        except:
            pass

    for i in sorted(title_dic.items(), key=lambda x: x[1], reverse=True):
        try:
            i = i[0]

            title = title_id[i]
            doc_lists.append((i, title))

        except:
            pass

    return doc_lists


def get_candidate_documents_and_scores(query, index, file_name):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """

    DL = DL_text
    words = list(index.term_total.keys())
    tokens = tokenize(query)
    query_counter = Counter(tokens)
    query_counterd = dict(query_counter)
    candidates = {}
    N = len(DL)
    for term in np.unique(tokens):

        if term in words:

            list_of_doc = read_posting_list(index, term, file_name)
            normlized_tfidf = []
            for doc_id, freq in list_of_doc:
                if (doc_id, freq) == (0, 0):
                    continue

                formula = (freq / DL[doc_id]) * math.log(N / index.df[term], 10) * query_counterd[term]
                id_tfidf = (doc_id, formula)
                normlized_tfidf.append(id_tfidf)

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def cosine_similarity(search_query, index, file_name):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    #########################################################

    dict_cosine_sim = {}
    candidates = get_candidate_documents_and_scores(search_query, index, file_name)
    for doc_id_term, normalized_tfidf in candidates.items():
        dict_cosine_sim[doc_id_term[0]] = normalized_tfidf / (len(search_query) * DL_text[doc_id_term[0]])

    return dict_cosine_sim


def search_body(query, index, file_name):
    doc_lists = []
    body_dict = cosine_similarity(query, index, file_name)
    for i in sorted(body_dict.items(), key=lambda x: x[1], reverse=True):
        try:
            i = i[0]

            title = title_id[i]
            doc_lists.append((i, title))

        except:
            pass

    return doc_lists[:100]


def search_anchor(query, index, file_name):
    query_to = tokenize(query)
    query_counter = Counter(query_to)
    words_counter = Counter()
    anchor_dic = {}
    doc_list_anchor = []
    for token in query_counter.keys():
        try:
            posting = read_posting_list(index, token, file_name)
            for doc_id, frequency in posting:
                if doc_id not in anchor_dic:
                    anchor_dic[doc_id] = 0
                anchor_dic[doc_id] += 1
        except:
            pass

    for i in sorted(anchor_dic.items(), key=lambda x: x[1], reverse=True):
        try:
            i = i[0]
            title = title_id[i]
            doc_list_anchor.append((i, title))
        except:
            pass

    return doc_list_anchor

def Search_BM25_with_condicion(queries):
    size_corpus = 6348910   #size of the corpus
    sum_sc = 2028630613    # sum of the length of all documents in the corpus
    avg_doc_length_of_all_corpus = sum_sc / size_corpus
    tokens = tokenize(queries)
    relevant_docs = []
    term_docid_freq = {}
    k1 = 1.5 # k1 is a tuning parameter of BM25
    b = 0.75 # b is a tuning parameter of BM25
    term_docid_freq = {}
    all_docs_distinct = set()

    for term in tokens:
        if term in body_index.term_total:
            list_docid_tf_foreach_term = read_posting_list(body_index, term,file_name_body)
            for doc_id, freq in list_docid_tf_foreach_term:
                if freq <= 50: # the condition
                  continue
                term_docid_freq[(term, doc_id)] = freq
                all_docs_distinct.add(doc_id)


    def BM25_score_docid_query(query, doc_id):
        idf = calc_idf(query)
        bm25 = 0
        for term in query:
            if (term, doc_id) in term_docid_freq:
                freq = term_docid_freq[(term, doc_id)]
                tf_weight = (k1 + 1) * freq
                idf_weight = query[term] * idf[term]
                doc_length_weight = freq + k1 * (1 - b + b * (body_index.DL[doc_id] / avg_doc_length_of_all_corpus))
                bm25 += idf_weight * (tf_weight / doc_length_weight)
        return bm25

    def calc_idf(query):
        idf = {}
        for term in query:
            if term in body_index.df.keys():
                term_in_doc = body_index.df[term]
                idf[term] = math.log(1 + (size_corpus - term_in_doc + 0.5) / (term_in_doc + 0.5)) 
            else:
                pass
        return idf


    doc_id_bm25 = [(doc_id, BM25_score_docid_query(dict(Counter(tokens)), doc_id)) for doc_id in all_docs_distinct] 
    doc_id_bm25=sorted(doc_id_bm25, key=lambda x: x[1], reverse=True)[:100] # take the top 100
    res = list({doc_id: title_id[doc_id] for doc_id, score in doc_id_bm25}.items()) # match doc_id to title

    return res
if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)


