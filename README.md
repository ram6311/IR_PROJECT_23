# IR_PROJECT_23

In this undertaking, we crafted an information retrieval engine utilizing the entirety of Wikipedia files as a component
of our culminating project in the "Information Retrieval" course. Our engine endeavors to locate the most pertinent Wikipedia documents
in relation to a given query. During the preprocessing phase, we availed ourselves of the aid of Google Cloud Storage (GCP) 
and other prevalent Python libraries such as pandas and numpy, among others.

# Project files:
* search_frontend.py contains the following functions:
    1. search_body(query, index, file_name) - search the body of the document
    2. search_anchor(query, index, file_name) - search the anchor of the document
    3. Search_BM25_with_condicion(queries) - search the body of the document with a condition
    
* BMֹ_25_from_index: class object that implement an index based on BM 25 score.

* inverted_index_gcp\colab is a class that contains the inverted index and the posting lists

* indexing_the_corpus_last_updated: we have devised three distinct inverted indexes, each one predicated on a specific aspect of 
    the Wikipedia documents (title, body, and anchor text). To achieve this, we employed Apache Spark for efficient processing and indexing.

* output.txt: contain bins location and load 

* Evaluations: In this file, we divided 30 queries into train and test sets. Subsequently, we evaluated 11 different versions of our engine by
    measuring their MAP@40 and average retrieval time. Following the training phase, we selected the optimal 
      version and applied it to the test set for further validation.
      
# NOTE
all this project is written in Python and implemented using PyCharm, JupyterNoteBook and GoogleColab.
