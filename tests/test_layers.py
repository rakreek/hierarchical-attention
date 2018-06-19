#  Run via "nosetests --nologcapture" to suppress all the chatty tensorflow stuff
import unittest
import numpy as np
np.random.seed(1357)
import numpy.testing as npt
import keras
from keras import backend as K
from keras.layers import Embedding, GRU, Bidirectional, Dense, Input
import hierarchical_attention.layers as han

class TestAttentionLayer(unittest.TestCase):

    def setUp(self):   
        self.x = np.array([[[-1, 1], [-2, 1], [1, -1.5]]])
        self.W = np.array([[1, 0], [1, 1]])
        self.b = np.array([1, 0])
        self.u = np.array([0.5, 0.1])
        self.v = np.array([[0.45695649,  0.07615942,  0.14054375]])

        self.attention_layer = han.AttentionLayer()
        self.attention_layer.build(self.x.shape)

    def test_call(self):
        expected = np.exp(self.v)/np.sum(np.exp(self.v))
        self.attention_layer.set_weights([self.W,self.b,self.u])
        actual = self.attention_layer.call(K.variable(self.x))
        npt.assert_almost_equal(K.eval(actual), expected)

    def test_call_u_0(self):
        expected = np.array([[1/3, 1/3, 1/3]])
        u = np.zeros(self.u.shape)
        self.attention_layer.set_weights([self.W, self.b, u])
        actual = self.attention_layer.call(K.variable(self.x))
        npt.assert_almost_equal(K.eval(actual), expected, decimal=6)

    def test_call_W_0(self):
        expected = np.array([[1/3, 1/3, 1/3]])
        W = np.zeros(self.W.shape)
        self.attention_layer.set_weights([W, self.b, self.u])
        actual = self.attention_layer.call(K.variable(self.x))
        npt.assert_almost_equal(K.eval(actual), expected, decimal=6)

    def test_call_with_mask(self):
        x = np.array([[[-1, 1], [-2, 1], [1, -1.5], [2, 5]]])
        mask = K.variable(np.array([[True, True, True, False]]))
        unmasked = np.exp(self.v)/np.sum(np.exp(self.v))
        expected = np.array([np.append(unmasked, np.array([0]))])
        self.attention_layer.set_weights([self.W, self.b, self.u])
        actual = self.attention_layer.call(K.variable(x), mask=mask)
        npt.assert_almost_equal(K.eval(actual), expected, decimal=6)

    def test_call_completely_masked_sequence(self):
        mask = K.variable(np.array([[False]*self.x.shape[1]]))
        actual = self.attention_layer.call(K.variable(self.x), mask=mask)
        expected = np.zeros(self.x.shape[:-1])
        npt.assert_almost_equal(K.eval(actual), expected, decimal=6)

    def test_compute_output_shape(self):
        expected = (1, 3,)
        x = K.variable(self.x)
        actual = self.attention_layer.compute_output_shape(x.shape)
        self.assertEqual(actual, expected)

    def test_shape_attention_matrix(self):
        batch_size = 8
        sent_len = 2
        embed_dim = 5
        attention_layer = han.AttentionLayer()
        attention_layer.build((batch_size, sent_len, embed_dim))
        actual = [w.shape for w in attention_layer.get_weights()]
        expected = [(embed_dim, embed_dim), (embed_dim,), (embed_dim,)]
        self.assertEqual(actual, expected)

class TestWeightedAverage(unittest.TestCase):

    def setUp(self):
        self.weighted_average = han.WeightedAverage()
        self.weights = K.variable(
            np.array(
                [[0.1, 0.7, 0.2],
                 [0.0, 0.0, 0.0],
                 [0.5, 0.0, 0.5],
                 [0.0, 1.0, 0.0]]
                )
            )
        self.vectors = K.variable(
            np.array(
                [[[-1, 0, 1], [1, 1, 0], [0,-1, 1]],
                 [[ 0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 [[-1, 0,-1], [0, 0, 0], [0,-1, 0]],
                 [[ 0, 0, 0], [1, 1, 1], [0, 0, 0]]]
                )
            )
        self.inputs = [self.weights, self.vectors]

    def test_compute_output_shape(self):
        expected = (4,3,)
        actual = self.weighted_average.compute_output_shape([self.weights.shape, self.vectors.shape])
        self.assertEqual(expected, actual)

    def test_compute_mask_none(self):
        actual = self.weighted_average.compute_mask(self.inputs, mask=None)
        self.assertIsNone(actual)

    def test_compute_mask_vector_mask_none(self):
        actual = self.weighted_average.compute_mask(self.inputs, mask=[None, None])
        self.assertIsNone(actual)

    def test_compute_mask(self):
        mask = K.variable(
            np.array(
                [[True , True , True ],
                 [False, False, False],
                 [True , False, True ],
                 [False, True , False]]
                )
            )
        expected = np.array([True, False, True, True])
        actual = K.eval(
            self.weighted_average.compute_mask(self.inputs, [None, mask])
            )
        npt.assert_array_equal(actual, expected)


    def test_call(self):
        expected = np.array([[ 0.6, 0.5, 0.3],
                             [ 0.0, 0.0, 0.0],
                             [-0.5,-0.5,-0.5],
                             [ 1.0, 1.0, 1.0]], dtype=np.float32)
        actual = K.eval(self.weighted_average.call(self.inputs))
        npt.assert_almost_equal(actual, expected)

class TestTimeDistributedWithMasking(unittest.TestCase):

    def setUp(self):
        self.corpus = np.array(
           [
             [
               [0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5]
             ],
             [
               [0, 0, 0, 0, 0],
               [3, 4, 5, 6, 7]
             ],
             [
               [0, 2, 4, 6, 7],
               [7, 0, 6, 1, 2]
             ]
           ], dtype=np.int32)


    def test_masked_values(self):
        model = keras.models.Sequential()
        td_layer = han.TimeDistributedWithMasking
        embedding = keras.layers.Embedding
        model.add(td_layer(embedding(8, 4, mask_zero=True),
                             input_shape=(2, 5),
                             name='td',
                             ))
        actual = K.eval(model.get_layer('td').compute_mask(K.variable(self.corpus)))
        expected = np.array([
                              [
                                [False, True, True, True, True],
                                [True, True, True, True, True],
                              ],
                              [
                                [False, False, False, False, False],
                                [True, True, True, True, True],
                              ],
                              [
                                [False, True, True, True, True],
                                [True, False, True, True, True]
                              ]
                            ])
        npt.assert_array_equal(actual, expected)