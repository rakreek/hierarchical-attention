# Hierarchical Attention

A [Keras](https://keras.io) implementation of ["Hierarchical Attention Networks for Document Classification"](http://www.aclweb.org/anthology/N16-1174) by Yang et al. 2016 (HAN). As implemented, the HAN is a classification model, but it could be adapted for other purposes. 

The main features of the network are a hierarchical representation of documents (i.e. documents are composed of sentences, which are composed of tokens), and an attention mechanism that assigns an attention weight to each sentence in the document, as well as each word in each sentence. The attention weights indicate the extent to which each word in a sentence is relevant to the vector representation of that sentence and the extent to which each sentence is relevant to the model's document-level prediciton.

## Example

``` python
import hierarchical_attention as han

# x is an nd-array of documents with shape (`n_docs`, `doc_len`, `sent_len`)
# y is an nd-array of labels with shape (`n_docs`, `n_labels`)

model = han.YangEtAl2016(
            sent_len=10, # 10 words per sentence
            doc_len=15, # 15 sentences per document
            vocab_size=100, # number of unique words in the input corpus
            word_embed_dim=200, # size of word vectors
            gru_dim=50, # size of hidden recurrent layers
            n_classes=2) # number of target output classes

model.fit(x, y) # trains the model
model.td_word_attention(x) # gets word attention weights for docs in x
model.sentence_attention(x) # gets sentence attention weights for docs in x
```

## Citation

```
@InProceedings{yang-EtAl:2016:N16-13,
  author    = {Yang, Zichao  and  Yang, Diyi  and  Dyer, Chris  and  He, Xiaodong  and  Smola, Alex  and  Hovy, Eduard},
  title     = {Hierarchical Attention Networks for Document Classification},
  booktitle = {Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2016},
  address   = {San Diego, California},
  publisher = {Association for Computational Linguistics},
  pages     = {1480--1489},
  url       = {http://www.aclweb.org/anthology/N16-1174}
}
```