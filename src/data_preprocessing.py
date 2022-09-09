'''
Generate processed data: filtering, combining ...
'''

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import collections
import os

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import category_encoders as ce
from sklearn.pipeline import Pipeline
import datetime
import json
import pickle
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

# from nltk import word_tokenize
# from nltk.stem import PorterStemmer, WordNetLemmatizer
#
# import scipy.io as sio
# class LemmaTokenizer:
#     def __init__(self):
#         self.wnl = WordNetLemmatizer()
#     def __call__(self, doc):
#         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def draw_bar(x, y, x_label, y_label=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def draw_freq(data, x_label=None, bool_discrete = False):
    fig = plt.figure()
    plt.hist(data, bins=50)

    plt.xlabel(x_label)
    plt.ylabel("Frequency")

    ax = fig.add_subplot(1, 1, 1)

    # Find at most 10 ticks on the y-axis
    if not bool_discrete:
        max_xticks = 10
        xloc = plt.MaxNLocator(max_xticks)
        ax.xaxis.set_major_locator(xloc)

    plt.show()

def filter_goodreads(path, save_flag=True):
    '''
    : filter: 1. review>=3; 2. author's book number >= 2 and <=50
    '''

    # read review/rating data
    rating_dict = {}
    review_num_dict = {}
    book_authors_dict = {}
    authors_book_dict = {}
    with open(path) as f:
        for line in f:
            data_line = json.loads(line)
            asin = data_line['isbn']
            if asin == "":
                continue
            if asin in rating_dict:
                print('repeated books!')

            rating_dict[asin] = float(data_line['average_rating'])
            review_num_dict[asin] = int(data_line['text_reviews_count'])
            book_authors_dict[asin] = data_line['authors']
            for author_info in data_line['authors']:
                author = author_info['author_id']
                if author in authors_book_dict:
                    authors_book_dict[author].append(asin)
                else:
                    authors_book_dict[author] = [asin]

    # filter: review>=3, author's book number >= 2
    print('all books: ', len(rating_dict))
    books_select = set([asin for asin in review_num_dict if review_num_dict[asin] >= 3])
    print('books with review >= 3: ', len(books_select))

    authors_booknum_dict = {author: len([book for book in authors_book_dict[author] if book in books_select]) for author
                            in authors_book_dict}  # all authors' number of books (books are filtered)

    # filter: author's book number >= 2
    books_select_2 = []
    for book in books_select:
        for author_info in book_authors_dict[book]:
            author = author_info['author_id']
            if authors_booknum_dict[author] >= 2 and authors_booknum_dict[author] <= 50:
                books_select_2.append(book)
                break

    authors_select = [author for author in authors_book_dict if
                      len(authors_book_dict[author]) > 2 and len(authors_book_dict[author]) <= 50]
    print('real hyperedges: authors with book > 2 and <= 50: ', len(authors_select))
    authors_select = set([author for author in authors_book_dict if
                          len(authors_book_dict[author]) >= 2 and len(authors_book_dict[author]) <= 50])
    print('hyperedges:  authors with book >= 2 and <=50: ', len(authors_select))

    # book_select_2 = set(books_select_2)
    print('books with authors who wrote books num >= 2: ', len(books_select_2))

    books_select = books_select.intersection(books_select_2)
    print('books selected: ', len(books_select))

    authors_book_dict = {author: list(set(authors_book_dict[author]).intersection(books_select)) for author in
                         authors_book_dict}  # update with select books

    max_hyperedge_size = 0
    for author in authors_select:
        if len(authors_book_dict[author]) > max_hyperedge_size:
            max_hyperedge_size = len(authors_book_dict[author])
    print('max_hyperedge_size: ', max_hyperedge_size)

    max_degree = 0
    book_authors_num_dict = {
        book: len([author['author_id'] for author in book_authors_dict[book] if author['author_id'] in authors_select])
        for book in book_authors_dict}  # all books' authors (authors are filtered)
    for book in books_select:
        if book_authors_num_dict[book] > max_degree:
            max_degree = book_authors_num_dict[book]
    print('max_degree: ', max_degree)

    # ratings
    all_ratings = np.array([int(rating_dict[asin]) for asin in books_select])
    unique, frequency = np.unique(all_ratings,
                                  return_counts=True)
    sort_index = np.argsort(frequency)[::-1]
    unique = unique[sort_index]
    frequency = frequency[sort_index]

    for k, v in zip(unique, frequency):
        print('rating: ', k, v)

    books_select = list(books_select)
    books_select.sort()
    authors_select = list(authors_select)
    authors_select.sort()
    # save into files
    if save_flag:
        data_save = {'books_select': books_select, 'authors_select': authors_select}
        with open('../data/goodreads_select.pickle', 'wb') as f:
            pickle.dump(data_save, f)
    return books_select, authors_select

def load_goodreads_select_meta(path, books_select, authors_select, save_flag=False):
    books_select_set = set(books_select)
    authors_select_set = set(authors_select)

    # read data
    rating_dict = {}
    book_authors_dict = {}  # book asin: [author1_ID, author2_ID...]
    authors_book_dict = {}
    book_descriptions = {}
    book_title = {}
    book_review_count = {}

    with open(path) as f:
        for line in f:
            data_line = json.loads(line)
            asin = data_line['isbn']
            if asin == "" or asin not in books_select_set:  # only focus on selected books!
                continue
            if asin in rating_dict:
                print('repeated books!')

            rating_dict[asin] = float(data_line['average_rating'])
            book_authors_dict[asin] = []
            for author_info in data_line['authors']:
                author = author_info['author_id']
                if author in authors_select_set:  # only focus on selected authors
                    book_authors_dict[asin].append(author)

                    if author in authors_book_dict:
                        authors_book_dict[author].append(asin)
                    else:
                        authors_book_dict[author] = [asin]

            book_descriptions[asin] = data_line['description'] + ' ' + data_line['title']
            book_title[asin] = data_line['title']
            book_review_count[asin] = int(data_line['text_reviews_count'])

    author_st = []
    title_st = []
    review_count_st = []

    for i in range(len(books_select)):
        author_st.append(book_authors_dict[books_select[i]])
        title_st.append(book_title[books_select[i]])
        review_count_st.append(book_review_count[books_select[i]])
    data_meta = {'title': title_st, 'authors': author_st, 'review_count': np.array(review_count_st)}
    return data_meta



def load_goodreads_select(path, books_select, authors_select, save_flag=False):
    books_select_set = set(books_select)
    authors_select_set = set(authors_select)

    # read data
    rating_dict = {}
    book_authors_dict = {}  # book asin: [author1_ID, author2_ID...]
    authors_book_dict = {}
    book_descriptions = {}

    with open(path) as f:
        for line in f:
            data_line = json.loads(line)
            asin = data_line['isbn']
            if asin == "" or asin not in books_select_set:  # only focus on selected books
                continue
            if asin in rating_dict:
                print('repeated books!')

            rating_dict[asin] = float(data_line['average_rating'])
            book_authors_dict[asin] = []
            for author_info in data_line['authors']:
                author = author_info['author_id']
                if author in authors_select_set:  # only focus on selected authors
                    book_authors_dict[asin].append(author)

                    if author in authors_book_dict:
                        authors_book_dict[author].append(asin)
                    else:
                        authors_book_dict[author] = [asin]

            book_descriptions[asin] = data_line['description'] + ' ' + data_line['title']

    # authors who have at least one books in book_select
    author_with_books = []
    for author in authors_select:
        if author in authors_book_dict and len(authors_book_dict[author])>1:  # authors with at least one book
            author_with_books.append(author)
    print('size: ', len(author_with_books))
    authors_select = author_with_books

    # bag of words
    corpus = [book_descriptions[asin] for asin in books_select]
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), max_features=500)  # tokenizer=LemmaTokenizer()
    cv_fit = cv.fit_transform(corpus)  # top 500 x word num
    word_name = cv.get_feature_names()  # dictionary
    print('word num: ', len(word_name), ' ', word_name)
    # normalize
    cv_fit = cv_fit.toarray()
    features = preprocessing.normalize(cv_fit)
    features += np.random.normal(0, 1, size=(features.shape[0], features.shape[1]))
    print('feature mean/std: ', np.mean(features), np.std(features))

    # treatment
    treatment = np.array([int(rating_dict[asin]) for asin in books_select])
    treatment[np.where(treatment <=3)] = 0
    treatment[np.where(treatment > 3)] = 1
    print('t=0 and t=1: ', (treatment==0).sum(), (treatment==1).sum())

    # hypergraph
    book2idx = {books_select[id]: id for id in range(len(books_select))}
    edge_idx_node = []
    edge_idx_edge = []
    for aid in range(len(authors_select)):
        books_cur = []
        for book in authors_book_dict[authors_select[aid]]:
            if book in book2idx:
                books_cur.append(book2idx[book])
        edge_idx_node = edge_idx_node + books_cur
        edge_idx_edge = edge_idx_edge + [aid for i in range(len(books_cur))]

    hyperedge_index = np.array([edge_idx_node, edge_idx_edge])


    data_save = {'features': features, 'treatment': treatment, 'hyper_index': hyperedge_index}

    if save_flag:
        with open('../data/goodreads_processed.pickle', 'wb') as f:
            pickle.dump(data_save, f)
    return data_save

def preprocess_goodreads(path):
    # books_select, authors_select = filter_goodreads(path, False)
    with open('../data/goodreads_select.pickle', 'rb') as f:
        data_select = pickle.load(f)
    books_select, authors_select = data_select['books_select'], data_select['authors_select']

    # load book features
    load_goodreads_select(path, books_select, authors_select, True)

    return

def hypergraph_stats(hyperedge_index, n):
    # hyperedge size
    unique_edge, counts_edge = np.unique(hyperedge_index[1], return_counts=True)  # edgeid, size
    ave_hyperedge_size = np.mean(counts_edge)
    max_hyperedge_size = np.max(counts_edge)
    min_hyperedge_size = np.min(counts_edge)
    m = len(unique_edge)

    sz, ct = np.unique(counts_edge, return_counts=True)  # hyperedgesize, count
    counts_edge_2 = ct[np.where(sz==2)][0]

    # node degree
    unique_node, counts_node = np.unique(hyperedge_index[0], return_counts=True)  # nodeid, degree
    ave_degree = np.mean(counts_node)
    max_degree = np.max(counts_node)
    min_degree = np.min(counts_node)
    statistics = {'n': n, 'm': m, 'm>2': m-counts_edge_2,
                  'average_hyperedge_size': ave_hyperedge_size, 'min_hyperedge_size': min_hyperedge_size, 'max_hyperedge_size': max_hyperedge_size,
                  'average_degree': ave_degree, 'max_degree': max_degree, 'min_degree': min_degree}
    return statistics

def preprocess_contact(path_root):
    path_nverts = path_root+'contact-high-school-nverts.txt'
    path_simplices = path_root+'contact-high-school-simplices.txt'

    # size of each hyperedge
    with open(path_nverts) as f:
        sizeOfEdge = f.readlines()
    f.close()
    sizeOfEdge = [int(i) for i in sizeOfEdge]
    m = len(sizeOfEdge)

    idx_start = []
    sum_size = 0
    for i in range(m):
        idx_start.append(sum_size)
        sum_size += sizeOfEdge[i]

    # nodes in each hyperedge
    with open(path_simplices) as f:
        edge_idx_node = f.readlines()
    f.close()
    edge_idx_node = [int(i) for i in edge_idx_node]
    # edge_idx_edge = [i for i in range(m) for j in range(sizeOfEdge[i])]

    # remove redundant hyperedges
    unique_edges = {}
    edge_idx_node_unique = []
    edge_idx_edge_unique = []
    for i in range(m):
        key_nodes = edge_idx_node[idx_start[i]: idx_start[i] + sizeOfEdge[i]]
        key_nodes.sort()
        key = ''
        for k in key_nodes:
            key += '_'+str(k)
        if key not in unique_edges:
            edge_idx_edge_unique += [len(unique_edges) for j in range(sizeOfEdge[i])]
            edge_idx_node_unique += edge_idx_node[idx_start[i]: idx_start[i] + sizeOfEdge[i]]
            unique_edges[key] = 1

    edge_idx_node_unique = [i-1 for i in edge_idx_node_unique]  # start from 0
    hyperedge_index = np.array([edge_idx_node_unique, edge_idx_edge_unique])

    # statistics
    n = np.max(hyperedge_index[0]) + 1
    statistics = hypergraph_stats(hyperedge_index, n)
    print(statistics)

    # record_data
    data_save = {'hyper_index': hyperedge_index}

    save_flag = True
    if save_flag:
        with open('../data/contact_hypergraph.pickle', 'wb') as f:
            pickle.dump(data_save, f)
    return data_save


if __name__ == '__main__':
    dataset = 'contact'
    if dataset == 'GoodReads':
        path = '../data/goodreads_books_children.json'
        preprocess_goodreads(path)
    elif dataset == 'contact':
        path_root = '../data/contact-high-school/'
        preprocess_contact(path_root)






