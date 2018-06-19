#  Run via "nosetests --nologcapture" to suppress all the chatty tensorflow stuff
import unittest
import numpy as np
import random
np.random.seed(1004)
random.seed(198705)
import numpy.testing as npt

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, GRU, Bidirectional, Dense, GlobalAveragePooling1D, Dot
from fntk import hierarchicalattention as han

# class IntegrationTestAttention(unittest.TestCase):

#     def test_attention_end_to_end(self):
#         vocab_size = 10
#         seq_len = 4
#         embed_dim = 3
#         n_samples = 10000
#         key_words = [1,2,]
#         x = np.array([np.random.choice(vocab_size, size=(seq_len,), replace=False) for _ in range(n_samples)])
#         y = np.isin(x, key_words).any(axis=1).astype(np.float)
#         x_test = np.array([[4, 0, 1, 6],
#                            [8, 4, 9, 3],
#                            [1, 4, 5, 0],
#                            [3, 7, 9, 2],
#                            [1, 8, 9, 3]])
#         y_test = np.array([1,0,1,1,1])
#         expected_attention = np.array([[0.  , 0.  , 0.99, 0.  ],
#                                        [0.25, 0.25, 0.25, 0.25],
#                                        [0.99, 0.  , 0.  , 0.  ],
#                                        [0.  , 0.  , 0.  , 0.99],
#                                        [0.99, 0.  , 0.  , 0.  ]])
        
#         embed_layer = Embedding(vocab_size,
#                                    embed_dim,
#                                    input_length=seq_len,
#                                    mask_zero=True,
#                                    name='embed')
#         attention_layer = han.AttentionLayer(name='attn')

#         input_layer = Input(shape=(seq_len,))
#         word_embedding = embed_layer(input_layer)
#         attention_weights = attention_layer(word_embedding)
#         avg_embedding = Dot(axes=1)([attention_weights, word_embedding])
#         output = Dense(1, activation='sigmoid')(avg_embedding)
#         model = Model(inputs=input_layer, outputs=output)

#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         model.fit(x, y, epochs=5, verbose=-1)
        
#         # Test Predictions
#         predicted_classes = (model.predict(x_test) > 0.5).reshape((-1,))
#         npt.assert_array_equal(predicted_classes, y_test)

#         # Test Attention Weights
#         x_test = K.variable(x_test)
#         embed_sents = embed_layer.call(x_test)
#         mask = embed_layer.compute_mask(x_test)
#         actual_attention = K.eval(attention_layer.call(embed_sents, mask))
#         npt.assert_almost_equal(actual_attention, expected_attention, decimal=2)

# class TestBiGRU(unittest.TestCase):

#     def setUp(self):
#         self.seq_len = 3
#         self.pad_len = 3
#         self.vocab_size = 5
#         self.embed_dim = 4
#         self.h_dim = 1

#         self.model = Sequential()
#         self.model.add(Embedding(self.vocab_size,
#                                  self.embed_dim,
#                                  input_length=self.seq_len,))
#         self.model.add(Bidirectional(GRU(self.h_dim)))

#         embed_wts = self.model.layers[0].get_weights()
#         gru_wts = self.model.layers[1].get_weights()

#         self.padded_model = Sequential()
#         self.padded_model.add(Embedding(self.vocab_size,
#                                  self.embed_dim,
#                                  input_length=self.seq_len + self.pad_len,
#                                  mask_zero=True,
#                                  weights=embed_wts))
#         self.padded_model.add(Bidirectional(GRU(self.h_dim), weights='foo'))
#         self.padded_model.layers[1].set_weights(gru_wts)

#     def test_masking_fwd_pass(self):
#         padded_x = np.array([[0,1,2,0,3,0]])
#         x = np.array([[1,2,3]])
#         padded_out = self.padded_model.predict(padded_x)
#         out = self.model.predict(x)
#         npt.assert_almost_equal(out, padded_out, decimal=5)

#     def test_masking_backward_pass(self):
#         # Test that the weight update for a padded, masked sequence is the same
#         # for a non-padded sequence of the same effectie length.
#         padded_x = np.array([[0,1,2,0,3,0]])
#         x = np.array([[1,2,3]])
#         y = np.array([10])
        
#         self.model.add(Dense(1))
#         self.model.compile(optimizer='sgd', loss='mse')
#         dense_wts = self.model.layers[2].get_weights()
#         self.padded_model.add(Dense(1, weights=dense_wts))
#         self.padded_model.compile(optimizer='sgd', loss='mse')

#         loss = self.model.train_on_batch(x, y)
#         padded_loss = self.padded_model.train_on_batch(padded_x, y)

#         wts = self.model.layers[1].get_weights()
#         padded_wts = self.padded_model.layers[1].get_weights()

#         self.assertEqual(loss, padded_loss)
#         for wt, padded_wt in zip(wts, padded_wts):
#             npt.assert_almost_equal(wt, padded_wt, decimal=6)

#     def test_time_distributed_and_masking_fwd_pass(self):        
#         pass

class TestHierarchicalAttentionNetwork(unittest.TestCase):

    def test_han(self):
        # conditions for positive class
          # sentence 1 has 1,2,3 in pos 0,1,2 and 5 in one of pos 6,7,8 (negated by 9).
          # sentence 5 has 1,2,3 in pos 
        # sentence 2 or 3 has 4,5,6 in the second third, and sentence 3 or 4 has 4,5,6 in the 3rd third
        # BUT if any of those is preceded by a 9, then not.
        



        # Params
        sent_len = 9
        doc_len = 5
        vocab_size = 10
        gru_dim = 3
        word_embed_dim = 3
        n_classes = 1
        n_samples = 120000

        # Synthetic Data


        x = np.random.choice(vocab_size, (n_samples, doc_len, sent_len))
        # Randomly set some sentences to 0 to simulate variable do lengths
        for i in np.random.choice(n_samples, int(.4*n_samples), replace=False):
            x[i, random.choice(range(1,6)):,:] = 0
        y = np.zeros(n_samples)
        for i in np.random.choice(n_samples, int(.75*n_samples), replace=False):
            s = random.choice([1,4])
            w = random.choice(range(5))
            w2 = random.choice([7,8])
            x[i, s, w:(w+3)] = [1,2,3]
            if random.random() > 0.2:
                x[i, s, w2] = 5
                y[i] = 1
                if random.random() > 0.6:
                    x[i, s, w2-1] = 9
                    y[i] = 0

        model = han.YangEtAl2016(
            vocab_size=vocab_size,
            sent_len=sent_len,
            doc_len=doc_len,
            word_embed_dim=word_embed_dim,
            gru_dim=gru_dim,
            n_classes=1)
        
        model.fit(x, y, epochs=3, validation_split=0.2)
        model.han.summary()
        print("inputs:\n", x[:5])
        print("actual:\n", y[:5])
        print("predicted:\n", np.round(model.han.predict(x[:5]), decimals=2))
        print("sentence attention:\n",
              np.round(model.sentence_attention.predict(x[:5]), decimals=2))
        print("word attentions\n",
              np.round(model.td_word_attention.predict(x[:5]), decimals=2))
        print("word vectors:\n",
              np.round(model.word_embedding_layer.get_weights(), decimals=3))