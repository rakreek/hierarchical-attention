#  Run via "nosetests --nologcapture" to suppress all the chatty tensorflow stuff
import unittest
import numpy as np
np.random.seed(1357)
import numpy.testing as npt
import keras
from keras import backend as K
import hierarchical_attention as han
import os

class TestYangEtAl2016(unittest.TestCase):
    
    def setUp(self):
        self.vocab_size=10
        self.word_embed_dim=6
        self.sent_len=9
        self.doc_len=5
        self.gru_dim=4
        self.n_classes = 1
        self.corpus = np.array(
            [
              [
                [0,1,2,3,0,1,2,3,4],
                [9,1,1,0,9,1,1,0,0],
                [6,6,6,5,5,5,4,4,4],
                [9,9,9,8,8,8,7,7,7],
                [0,0,0,0,0,0,0,0,0],
              ],
              [
                [4,3,2,1,0,5,6,7,8],
                [4,2,5,9,2,2,4,0,9],
                [4,2,5,7,4,6,3,6,5],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
              ],
              [
                [8,4,7,4,0,2,3,4,8],
                [3,3,3,2,2,2,1,1,1],
                [0,0,0,0,0,0,0,0,0],
                [0,5,2,8,1,9,8,7,0],
                [5,5,5,5,5,5,5,5,5],
              ],
            ],
            dtype=np.int32
        )
        self.model = han.YangEtAl2016(
            sent_len=self.sent_len,
            doc_len=self.doc_len,
            vocab_size=self.vocab_size,
            word_embed_dim=self.word_embed_dim,
            gru_dim=self.gru_dim,
            n_classes=self.n_classes)

    def test_td_weights_equal_wrapped_weights(self):
        self.model.build()
        td_layers = ['td_word_embeddings',
                     'td_word_bi_gru',
                     'td_word_attention',]
        wrapped_layers = [self.model.word_embedding_layer,
                          self.model.word_bi_gru_layer,
                          self.model.word_attention_layer,]
        td_weights = [self.model.han.get_layer(l).get_weights() for l in td_layers]
        weights = [l.get_weights() for l in wrapped_layers]
        for l1, l2 in zip(td_weights, weights):
            for w1, w2 in zip(l1, l2):
                npt.assert_array_equal(w1, w2)

    def test_td_word_attention(self):
        word_attention = self.model.td_word_attention(self.corpus)
        self.assertEqual(word_attention[0,0,0], 0.)
        self.assertEqual(word_attention[0,0,4], 0.)
        self.assertEqual(word_attention[0,1,3], 0.)
        self.assertAlmostEqual(word_attention[0,0].sum(), 1., places=6)
        self.assertAlmostEqual(word_attention[0,1].sum(), 1., places=6)
        self.assertAlmostEqual(word_attention[0,2].sum(), 1., places=6)

    def test_sentence_attention(self):
        word_attention = self.model.sentence_attention(self.corpus)
        self.assertEqual(word_attention[0,4], 0.)
        self.assertEqual(word_attention[1,3], 0.)
        self.assertEqual(word_attention[1,4], 0.)
        self.assertEqual(word_attention[2,2], 0.)
        self.assertAlmostEqual(word_attention[0].sum(), 1., places=6)
        self.assertAlmostEqual(word_attention[1].sum(), 1., places=6)
        self.assertAlmostEqual(word_attention[2].sum(), 1., places=6)

    def test_pretrained_word_vectors(self):
        pretrained_word_vectors = np.ones((self.vocab_size, self.word_embed_dim))
        model = han.YangEtAl2016(
            sent_len=self.sent_len,
            doc_len=self.doc_len,
            pretrained_word_vectors=pretrained_word_vectors,
            gru_dim=self.gru_dim,
            n_classes=self.n_classes)
        model.build()
        actual = model.word_embedding_layer.get_weights()
        actual_td = model.han.get_layer('td_word_embeddings').get_weights()
        npt.assert_array_equal(actual[0], pretrained_word_vectors)
        npt.assert_array_equal(actual_td[0], pretrained_word_vectors)

    def test_output_shapes_td_word_attention(self):
        self.model.build()
        actual = {l.get_config()['name']: l.output_shape for l in self.model._td_word_attention.layers}
        expected = {
            'han_inputs': (None, self.doc_len, self.sent_len),
            'td_word_embeddings': (None, self.doc_len, self.sent_len, self.word_embed_dim),
            'td_word_bi_gru': (None, self.doc_len, self.sent_len, 2 * self.gru_dim),
            'td_word_attention': (None, self.doc_len, self.sent_len),
            }
        self.assertEqual(actual, expected)

    def test_output_shapes_sentence_attention(self):
        self.model.build()
        actual = {l.get_config()['name']: l.output_shape for l in self.model._sentence_attention.layers}
        expected = {
            'han_inputs': (None, self.doc_len, self.sent_len),
            'td_word_embeddings': (None, self.doc_len, self.sent_len, self.word_embed_dim),
            'td_word_bi_gru': (None, self.doc_len, self.sent_len, 2 * self.gru_dim),
            'td_word_attention': (None, self.doc_len, self.sent_len),
            'sentence_vectors': (None, self.doc_len, 2 * self.gru_dim),
            'sentence_bi_gru': (None, self.doc_len, 2 * self.gru_dim),
            'sentence_attention': (None, self.doc_len),
            }
        self.assertEqual(actual, expected)

    def test_output_shapes_han(self):
        self.model.build()
        actual = {l.get_config()['name']: l.output_shape for l in self.model.han.layers}
        expected = {
            'han_inputs': (None, self.doc_len, self.sent_len),
            'td_word_embeddings': (None, self.doc_len, self.sent_len, self.word_embed_dim),
            'td_word_bi_gru': (None, self.doc_len, self.sent_len, 2 * self.gru_dim),
            'td_word_attention': (None, self.doc_len, self.sent_len),
            'sentence_vectors': (None, self.doc_len, 2 * self.gru_dim),
            'sentence_bi_gru': (None, self.doc_len, 2 * self.gru_dim),
            'sentence_attention': (None, self.doc_len),
            'document_embedding': (None, 2 * self.gru_dim),
            'document_output': (None, self.n_classes)}
        self.assertEqual(actual, expected)

    def test_masking_document_encoder(self):
        self.model.build()
        input_layer = self.model.han.input
        sentence_vectors = self.model.han.get_layer('sentence_vectors').output
        doc_enc = keras.models.Model(inputs=input_layer, outputs=sentence_vectors)
        actual = K.eval(doc_enc.compute_mask(K.variable(self.corpus), mask=None))
        expected = np.array(
              [
                [True, True, True, True, False,],
                [True, True, True, False, False,],
                [True, True, False, True, True,],
              ]
            )
        npt.assert_array_equal(actual, expected)
        
    def test_save_load_weights(self):
        unequal_weights = []
        self.model.build()
        model_layers = [l for l in dir(self.model) if l[-5:]=='layer']
        temp_dir = os.path.dirname(os.path.abspath(__file__))
        self.model.save_weights(os.path.join(temp_dir, 'temp_weights'))
        loaded_model = han.YangEtAl2016(
            sent_len=self.sent_len,
            doc_len=self.doc_len,
            vocab_size=self.vocab_size,
            word_embed_dim=self.word_embed_dim,
            gru_dim=self.gru_dim,
            n_classes=self.n_classes)
        loaded_model.build()
        loaded_model.load_weights(os.path.join(temp_dir, 'temp_weights'))
        os.remove(os.path.join(temp_dir, 'temp_weights'))
        for layer in model_layers:
            model_weights = eval('self.model.%s.get_weights()' % layer)
            loaded_weights = eval('loaded_model.%s.get_weights()' % layer)
            weights_diff=False
            for w1, w2 in zip(model_weights, loaded_weights):
                if not np.array_equal(w1, w2):
                    weights_diff = True
            if weights_diff:
                unequal_weights.append(layer)
        expected = []
        self.assertEqual(expected, unequal_weights)