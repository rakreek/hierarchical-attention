import glob
from nltk import word_tokenize,sent_tokenize
import io
from random import shuffle
import json
import multiprocessing
import gensim, logging
from datetime import datetime
import glob
import bs4
from xml.dom import minidom
from bs4 import BeautifulSoup
import io

PATHNAME = "/mnt/nfs/mirror/"
#PATHNAME = "/nas/qez/"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)



def text_to_tokens(text):
    sentences = sent_tokenize(text)
    tokens = []
    for s in sentences:
        tokens.extend([x.lower() for x in word_tokenize(s)])
    return tokens

def text_to_sentences(text):
    result = [x.lower() for x in sent_tokenize(text)]
    return result


def load20news():
    texts = []
    category = []
    for fname in glob.glob("%seapos/data/effects-of-noise/datasets/20news-18828/*/*" % PATHNAME):
        _, cat, _ = fname.rsplit("/", 2)
        text = io.open(fname, encoding="latin1").read()
        text=gensim.utils.any2unicode(text, encoding='latin1')
        texts.append(text_to_tokens(text))
        category.append(cat)
    return texts, category


def to_numeric_labels(categories):
    cat_to_numeric={}
    numeric_to_cat={}
    for i,cat in enumerate(set(categories)):
        numeric_to_cat[i]=cat
        cat_to_numeric[cat]=i
    return cat_to_numeric, numeric_to_cat


def process_yelp_json(x):
    json_obj = json.loads(x)
    return (text_to_tokens(json_obj['text']), json_obj['stars'])

def process_yelp_json_to_sentences(x):
    json_obj = json.loads(x)
    text = json_obj['text']
    return (text_to_sentences(text), json_obj['stars'])


def read_yelp(myfunc=process_yelp_json, max=None):
    return _read_yelp(myfunc=myfunc, max=max,
              fpath='%seapos/data/effects-of-noise/datasets/yelp_dataset/review.json' % PATHNAME)

def read_yelp2015(myfunc=process_yelp_json, max=None):
    return _read_yelp(myfunc=myfunc, max=max,
              fpath='%seapos/data/effects-of-noise/datasets/yelp_dataset/review2015.json' % PATHNAME)

def read_yelp2016(myfunc=process_yelp_json, max=None):
    return _read_yelp(myfunc=myfunc, max=max,
              fpath='%seapos/data/effects-of-noise/datasets/yelp_dataset/review2016.json' % PATHNAME)

def _read_yelp(myfunc=process_yelp_json, max=None,
              fpath='%seapos/data/effects-of-noise/datasets/yelp_dataset/review.json' % PATHNAME):
    f = open(fpath)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    lines=f.readlines()
    if max:
        lines=lines[:max]
    result=pool.map(myfunc,lines)
    return list(result)


def read_reuters21578():
    texts = []
    result_labels = []
    for f in glob.glob('%seapos/data/effects-of-noise/datasets/reuters21578/*.sgm'% PATHNAME):
        print(f)
        if f.endswith("reut2-017.sgm"):
            content = io.open(f, encoding="latin1").read()
        else:
            content=io.open(f, encoding="utf-8").read()
        soup=bs4.BeautifulSoup(content,'html.parser')
        reuters = soup.findAll("reuters")
        for reuter in reuters:
            topics=reuter.findAll('topics')
            labels=set()
            for topic in topics:
                categories=topic.findAll("d")
                labels=labels.union([x.text for x in categories])
                titles=reuter.findAll('title')
                titles = [x.text for x in titles]
                bodies = reuter.findAll('body')
                bodies = [x.text for x in bodies]
                text = ' '.join(titles+bodies)
                texts.append(text_to_tokens(text))
                result_labels.append(list(labels))
    return texts, result_labels


def load_farmads_texts():
    lines=open('%seapos/data/effects-of-noise/datasets/farm-ads/farm-ads.txt'% PATHNAME).readlines()
    texts=[text_to_tokens(gensim.utils.any2unicode(x.split(' ',1)[1])) for x in lines]
    return texts

def load_nsf_abstracts_texts(max):
    texts=[]
    for f in glob.glob('%seapos/data/effects-of-noise/datasets/nsfabs-mld/Part3/*/*/*.txt' % PATHNAME)[:max]:
        content=gensim.utils.any2unicode(io.open(f, encoding="latin1").read().strip(), encoding='latin1')
        texts.append(text_to_tokens(content))
    return texts

def load_legal_cases(max):
    texts=[]
    for f in glob.glob('%seapos/data/effects-of-noise/datasets/legal_cases/corpus/fulltext/*.xml' % PATHNAME)[:max]:
        soup = BeautifulSoup(io.open(f, encoding='utf-8').read())
        text=' '.join([x.text.strip().replace("\n", " ") for x in soup.findAll(['name','sentence'])])
        texts.append(text_to_tokens(gensim.utils.any2unicode(text)))
    return texts


def read_synthetic():
    print("reading reuters")
    texts_reuters, labels=read_reuters21578()
    print("reading 40k yelp")
    texts_yelp, labels=read_yelp2016(max=40000)
    print("reading 20 news")
    texts_20news, labels=load20news()
    print("reading farm ads")
    ads=load_farmads_texts()
    print("reading nfs")
    texts_nfs_abstracts=load_nsf_abstracts_texts(40000)
    texts=list(texts_reuters)+list(texts_yelp)+list(texts_20news)+list(ads)+list(texts_nfs_abstracts)
    labels=['reuters']*len(texts_reuters)+['yelp']*len(texts_yelp)+['20news']*len(texts_20news)\
           +['farmads']*len(ads)+['nfsabstract']*len(texts_nfs_abstracts)
    zipped=list(zip(texts, labels))
    shuffle(zipped)
    return list(zip(*zipped))

