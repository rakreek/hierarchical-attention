''' 
Utility functions for pre-processing data to feed to the hierarchical attention
model. These should probably wind up in FNTK at some point in the future. But by
the same token, some of them make assumptions about the structure of the data
(i.e. one document is a list of sentences which is a list of tokens) that will
not be valid in general.
'''

import gensim
import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter


def insert_unk_tokens(train_docs,
                      test_docs,
                      min_word_count=6,
                      unknown_token='unknown_token'):
    ''' 
    Replace rare words with an 'unknown_token'. In addition, replace words 
    appearing in the test set but not the train set with 'unknown_token'.  

    Arguments
    ---------
    train_docs : pandas.Series of lists of lists of tokens.
        Each item in the series represents a document, consisting of a list of
        sentences, each of which consists of a list of words.
    test_docs : pandas.Series of list of lists of tokens.
         Same logical structure as train_docs.
    min_word_count : `int` words with fewer occurrences are replaced with the
        'unknown token'.
    unkown_token : `str` 'unknown_token' by default. Change as needed.

    Returns
    -------
    train_docs_mod: train_docs with rare words replaced by 'unknown_token'.
    test_docs_mod: test_docs with rare train words and out-of-training-vocab words 
        replaced by 'unknown token'.
    '''

    train_wc = Counter()
    for doc in train_docs:
        for sentence in doc:
            train_wc.update(sentence)

    def replace_unk(doc):
        doc_mod = [
            [
                unknown_token if train_wc[word] < min_word_count else word
                for word in sent
            ]
            for sent in doc
        ]
        return doc_mod

    train_docs_mod = train_docs.apply(replace_unk)
    test_docs_mod = test_docs.apply(replace_unk)

    return train_docs_mod, test_docs_mod

def word_to_vec(corpus, embed_dim, pad_zero=True, **kwargs):
    """
    Train word vectors with `gensim.models.Word2Vec`

    Parameters
    ----------
    corpus : list of lists
        Corpus is a list of documents, which is in turn a list of sentences.
        A sentence is a list of words/tokens
    embed_dim : `int`
        Size of word-embedding dim.
    pad_zero : `bool`
        If True, prepend a word vector of all 0s for the padding token, and
        index vocab words starting with 1.
    kwargs : dict
        Additional keyword args to be passed to `gensim.Word2Vec`

    Returns
    -------
    weights : np.array of shape (`vocab_size`, `word_embed_dim`)
    vocab : dict
        keys are the original words/tokens
        values are the token's index, as int
    """

    class CorpusIterator(object):
        """Generator to serve up sentences to Word2Vec"""
        def __init__(self, corpus):
            self.corpus = corpus

        def __iter__(self):
            for doc in self.corpus:
                for sentence in doc:
                    yield sentence

    corpus_gen = CorpusIterator(corpus)        

    model = gensim.models.Word2Vec(corpus_gen, size=embed_dim, **kwargs)
    weights = model.wv.syn0

    index_shift = 0

    if pad_zero:
        weights = np.insert(weights, 0, 0, axis=0)
        index_shift = 1

    vocab = {k: (v.index + index_shift) for k, v in model.wv.vocab.items()}

    return weights, vocab

def indexize_doc(doc, vocab_dict):
    """
    Replace words in sentences with their word vector indexes

    Parameters
    ----------
    corpus : list of lists
        Corpus is list of sentences, each sentence is a list of words/tokens
    vocab_dict : dict
        keyed by word, value is index

    Returns
    -------
    corpus : list of lists
        Copy of corpus with the words replaced with their index

    See Also
    --------
    reverse_indexize_corpus

    """
    indexized_doc = [[vocab_dict[w] for w in sent] for sent in doc]
    return indexized_doc

def get_doc_and_sent_length(docs, doc_pct=1., sent_pct=1.):
    '''
    Find the n-th percentile document length and the m-th percentile sentence
    length for a corpus.  

    Arguments
    ---------
    docs : pd.Series of lists of lists of words/tokens
    doc_pct : float (default None)
        The percentile length to find. Defaults to max.
    sent_pct : float (default None)
        Percentile sentence length to find. Defaults to max.

    Returns
    -------
    doc_len : `int` n-th percentile document length
    sent_len : `int` n-th percentile of all sentence lengths
    '''
    assert isinstance(doc_pct, float), 'doc_pct must be a float.'
    assert isinstance(sent_pct, float), 'sent_pct must be a float.'

    doc_lens = docs.apply(len)
    doc_len = doc_lens.quantile(doc_pct, interpolation='lower')
    sent_lens = pd.Series(len(sent) for doc in docs for sent in doc)
    sent_len = sent_lens.quantile(sent_pct, interpolation='lower')

    return (doc_len, sent_len)

def docs_to_tensor(docs, vocab, doc_len, sent_len):
    '''
    Convert a batch (list) of documents to a tensor for training or scoring.
    
    Tokens are mapped to corresponding indices in a vocab dictionary. Sentences
    are padded with 0 or truncated to be of length sent_len. Documents are are
    padded with all-0 sentences or truncated to be of length doc_len. Words/
    sentences are dropped from the beginning of sentences/documents.

    Arguments
    ---------
    docs : list of documents, each of which is a list of sentences, each of
        which is a list of words/tokens.
    vocab: `dict`
    doc_len : `int`
    sent_len : `int`

    Returns
    -------
    docs_tensor: numpy array of shape (`len(docs)`, `doc_len`, `sent_len`)
    '''
    n_batch = len(docs)
    docs_tensor = np.zeros((n_batch, doc_len, sent_len,))
    for i, doc in enumerate(docs):
        doc = doc[-doc_len:]
        doc = indexize_doc(doc, vocab)
        for j, sent in enumerate(doc):
            if len(sent) < sent_len:
                docs_tensor[i, j, :len(sent)] = sent
            else:
                docs_tensor[i, j] = sent[-sent_len:]
            
    return docs_tensor