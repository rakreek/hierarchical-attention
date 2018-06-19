import numpy as np
import random
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Bidirectional, GRU
from keras.engine.topology import _object_list_uid
from keras.preprocessing.sequence import pad_sequences
import os, gensim
import math
try:
    from .layers import AttentionLayer, TimeDistributedWithMasking, WeightedAverage
except:
    from layers import AttentionLayer, TimeDistributedWithMasking, WeightedAverage



class YangEtAl2016(object):
    """ A hierarchical recurrent neural network model with attention based on [1]_

    For now, this only works with the TensorFlow backend.

    .. [1] Yang, Z.; Yang, D.; Dyer; He; Smola; Hovy. "Hierarchical Attention
    Networks for Document Classification", http://aclweb.org/anthology/N/N16/N16-1174.pdf

    Parameters
    ----------
    vocab_size : Number of unique tokens in the training vocabulary, including 
        the padding token.
        (Ignored if `pretrained_word_vectors` is not `None`.)
    word_embed_dim : size of word-embedding vectors
        (Ignored if `pretrained_word_vectors` is not `None`.)
    pretrained_word_vectors : Matrix of shape (`vocab_size`, `word_embed_dim`),
        containing pre-trained word-vector weights. The shape overwrites
        `vocab_size` and `word_embed_dim` args.
    doc_len : Number of sentences per input document. assumes all documents used
        to train or score will be padded/truncated to a standardized length. If
        `None`, then all documents within a batch must still be padded/truncated
        to the same length.
    sent_len : Number of words per sentence. All sentences in all docs used for
        training or scoring must be padded/truncated to this length.
    gru_dim : Size of output/hidden state of the GRU.
    n_classes : Number of output classes. (For binary target, used 1.)
    batch_size : Number of records in one batch.
    verbose :  Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch

    Attributes
    ----------
    han : The hierarchical attention network

    Methods
    -------
    build() : Build and compile the hierarchical attention model.
    fit(x, y, **kwargs) : Train the model.
    td_word_attention(docs) : Compute word-level attentions for each sentence in
        a batch of docs.
    sentence_attention(docs) : Compute sentence attentions for a batch of docs.
    summary(self, **kwargs) : Provide summary of the model architecture.
    predict(self, x, **kwargs): Predict the classifications for data x.
    predict_generator(self, generator, steps, **kwargs) : Predict classifications for batches yielded from generator.
    fit_generator(self, generator, steps_per_epoch, **kwargs) : Train the model on batches yieled from generator.
    save_weights(self, filepath, **kwargs) : Save the weights of trained model.
    """
    
    def __init__(self, vocab_size=None, word_embed_dim=200, sent_len=None,
                 doc_len=None, n_classes=None, gru_dim=50,
                 pretrained_word_vectors=None, batch_size=64, verbose=2):
        
        # Constants
        self.batch_size = batch_size
        self.optimizer = 'adam'
        self.metrics = ['accuracy']

        # Parameters
        ## Word Embedding
        if pretrained_word_vectors is not None:
            if not isinstance(pretrained_word_vectors, list):
                pretrained_word_vectors = [pretrained_word_vectors]
            vocab_size = pretrained_word_vectors[0].shape[0]
            word_embed_dim = pretrained_word_vectors[0].shape[1]
        self.vocab_size = vocab_size
        self.word_embed_dim = word_embed_dim
        self.sent_len = sent_len
        self.verbose = verbose
        ## Word-Level BiGRU
        self.gru_dim = gru_dim
        self.sentence_encoder_input_shape = (self.sent_len,)

        self.doc_len = doc_len
        self.han_input_shape = (self.doc_len, self.sent_len,)
        ## Output Layer
        if not isinstance(n_classes, int) or n_classes < 1:
            raise(ValueError, "`n_classes` must be a positive integer.")
        if n_classes == 1:
            self.output_activation = 'sigmoid'
            self.loss = 'binary_crossentropy'
        else:
            self.output_activation = 'softmax'
            self.loss = 'categorical_crossentropy'


        self.word_embedding_layer = Embedding(self.vocab_size, self.word_embed_dim,
                                              input_length=self.sent_len,
                                              mask_zero=True,
                                              name='word_embeddings',
                                              weights=pretrained_word_vectors)
        self.word_bi_gru_layer = Bidirectional(GRU(self.gru_dim, return_sequences=True), name='word_bi_gru')
        self.word_attention_layer = AttentionLayer(name='word_attention')
        self.sentence_bi_gru_layer = Bidirectional(GRU(self.gru_dim, return_sequences=True), name='sentence_bi_gru')
        self.sentence_attention_layer = AttentionLayer(name='sentence_attention')
        self.sentence_weighted_average_layer = WeightedAverage(name='document_embedding')
        self.output_layer = Dense(n_classes, activation=self.output_activation, name='document_output')
        self.td_word_embedding_layer = TimeDistributedWithMasking(self.word_embedding_layer, name='td_word_embeddings',
                                                                  weights=pretrained_word_vectors)
        self.td_word_bi_gru_layer = TimeDistributedWithMasking(self.word_bi_gru_layer, name='td_word_bi_gru')
        self.td_word_attention_layer = TimeDistributedWithMasking(self.word_attention_layer, name='td_word_attention')
        self.td_word_weighted_average_layer = WeightedAverage(name='sentence_vectors')

        # Models
        self._td_word_attention = None
        self._sentence_attention = None
        self.han = None

    def build(self):

        input_layer = Input(self.han_input_shape, name='han_inputs')
        td_word_embedding = self.td_word_embedding_layer(input_layer)
        td_word_bi_gru = self.td_word_bi_gru_layer(td_word_embedding)
        td_word_attention = self.td_word_attention_layer(td_word_bi_gru)
        sentence_vectors = self.td_word_weighted_average_layer([td_word_attention, td_word_bi_gru])
        sentence_bi_gru = self.sentence_bi_gru_layer(sentence_vectors)
        sentence_attention = self.sentence_attention_layer(sentence_bi_gru)
        document_embedding = self.sentence_weighted_average_layer([sentence_attention, sentence_bi_gru])
        document_output = self.output_layer(document_embedding)
        self._td_word_attention = Model(inputs=input_layer, outputs=td_word_attention)
        self._sentence_attention = Model(inputs=input_layer, outputs=sentence_attention)
        self.han = Model(inputs=input_layer, outputs=document_output)
        self.han.compile(self.optimizer, self.loss, metrics=self.metrics)
        self.layers = self.han.layers

    def fit(self, x, y, **kwargs):
        # TODO: Create a data generator that groups docs/sequences by length.
        #       Then, use fit_generator().
        if self.han is None:
            self.build()
        return self.han.fit(x=x, y=y, batch_size=self.batch_size, **kwargs)

    def td_word_attention(self, docs):
        '''
        Get word-level attention for each sentence in a batch of input documents.

        Arguments
        ---------
        docs : A numpy array with shape (`batch_size`, `doc_len`, `sent_len`).

        Returns
        -------
        A numpy array with the same shape as `docs`.
        '''
        if self._td_word_attention is None:
            self.build()
        return self._td_word_attention.predict(docs)

    def sentence_attention(self, docs):
        '''
        Get sentence-level attention for each sentence in a batch of input documents.

        Arguments
        ---------
        docs : A numpy array with shape (`batch_size`, `doc_len`, `sent_len`).

        Returns
        -------
        A numpy array with shape as (`batch_size`, `doc_len`).
        '''
        if self._sentence_attention is None:
            self.build()
        return self._sentence_attention.predict(docs)
    
    def summary(self, **kwargs):
        if self.han is None:
            self.build()
        return self.han.summary(**kwargs)
    
    def predict(self, x, **kwargs):
        if self.han is None:
            self.build()
        return self.han.predict(x=x, **kwargs)
    
    def predict_generator(self, generator, steps, **kwargs):
        if self.han is None:
            self.build()
        return self.han.predict_generator(generator=generator, steps=steps, **kwargs)
    
    def fit_generator(self, generator, steps_per_epoch, **kwargs):
        if self.han is None:
            self.build()
        return self.han.fit_generator(generator, steps_per_epoch, **kwargs)
    
    def save_weights(self, filepath, **kwargs):
        if self.han is None:
            self.build()
        return self.han.save_weights(filepath=filepath, **kwargs)
    
    def load_weights(self, filepath, **kwargs):
        if self.han is None:
            self.build()
        return self.han.load_weights(filepath=filepath, **kwargs)
