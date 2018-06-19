import unittest
import numpy as np
import pandas as pd
np.random.seed(2468)
import numpy.testing as npt
import hierarchical_attention.utils as utils

class TestInsertUnkTokens(unittest.TestCase):
    
    def setUp(self):
        self.train_docs = pd.Series([
                  [['a','a','a','a'],
                   ['a','b','b']],
                  [['b','b','b'],
                   ['c','c'],
                   ['d']]])
        self.test_docs = pd.Series([
                  [['a','a','a','a','a'],
                   ['a','b','b','b']],
                  [['b','e']]])

    def test_insert_unk_tokens(self):
        train_actual, \
        test_actual = utils.insert_unk_tokens(self.train_docs,
                                              self.test_docs,
                                              min_word_count=5)
        train_expected = pd.Series([
                  [['a','a','a','a'],
                   ['a','b','b']],
                  [['b','b','b'],
                   ['unknown_token','unknown_token'],
                   ['unknown_token']]])
        test_expected = pd.Series([
                  [['a','a','a','a','a'],
                   ['a','b','b','b']],
                  [['b','unknown_token']]])
        npt.assert_array_equal(train_actual.values, train_expected.values)
        npt.assert_array_equal(test_actual.values, test_expected.values)

    def test_custom_unknown_token(self):
        train_actual, \
        test_actual = utils.insert_unk_tokens(self.train_docs,
                                              self.test_docs,
                                              min_word_count=5,
                                              unknown_token='u')
        train_expected = pd.Series([
                  [['a','a','a','a'],
                   ['a','b','b']],
                  [['b','b','b'],
                   ['u','u'],
                   ['u']]])
        test_expected = pd.Series([
                  [['a','a','a','a','a'],
                   ['a','b','b','b']],
                  [['b','u']]])
        npt.assert_array_equal(train_actual.values, train_expected.values)
        npt.assert_array_equal(test_actual.values, test_expected.values)



class TestWordToVec(unittest.TestCase):
    
    def setUp(self):
        txt = r'''One fish Two fish Red fish Blue fish .
        Black fish Blue fish Old fish New fish .
        This one has a little star . This one has a little car . Say ! What a lot
        Of fish there are .
        Yes . Some are red . And some are blue . Some are old . And some are new .
        Some are sad .
        And some are glad .
        And some are very , very bad .
        Why are they Sad and glad and bad ? I do not know .
        Go ask your dad .
        Some are thin . And some are fat . The fat one has A yellow hat .
        From there to here , from here to there , Funny things
        Are everywhere .
        Here are some Who like to run . They run for fun In the hot , hot sun .
        Oh me ! Oh my !
        Oh me ! Oh my ! What a lot
        Of funny things go by .
        Some have two feet And some have four . Some have six feet And some have more .
        Where do they come from ? I can’t say . But I bet they have come a long , long way .
        We see them come . We see them go . Some are fast . And some are slow .
        Some are high And some are low . Not one of them Is like another . Don’t ask us why . Go ask your mother .
        Say !
        Look at his fingers ! One , two , three ... How many fingers Do I see ?
        One , two , three , four , Five , six , seven , Eight , nine , ten . He has eleven !
        Eleven !
        This is something new . I wish I had Eleven , too !'''
        self.txt = [[s.split()] for s in txt.lower().split('\n')]
        self.embed_dim = 10

    def test_weights_dim(self):
        weights, _ = utils.word_to_vec(self.txt,
                                       self.embed_dim,
                                       min_count=1)
        expected = (108, 10)
        self.assertEqual(weights.shape, expected)

    def test_vocab_size(self):
        _, vocab = utils.word_to_vec(self.txt,
                                     self.embed_dim,
                                     min_count=1)
        expected = 107
        self.assertEqual(len(vocab), expected)

    def test_min_index(self):
        _, vocab = utils.word_to_vec(self.txt,
                                     self.embed_dim,
                                     min_count=1)
        expected = 1
        self.assertEqual(min(vocab.values()), expected)

    def test_weights_dim_no_pad(self):
        weights, _ = utils.word_to_vec(self.txt,
                                       self.embed_dim,
                                       pad_zero=False,
                                       min_count=1)
        expected = (107, 10)
        self.assertEqual(weights.shape, expected)

    def test_min_index_no_pad(self):
        _, vocab = utils.word_to_vec(self.txt,
                                     self.embed_dim,
                                     pad_zero=False,
                                     min_count=1)
        expected = 0
        self.assertEqual(min(vocab.values()), expected)

class TestIndexizeDoc(unittest.TestCase):
    
    def test_indexize_doc(self):
        doc = [['one', 'fish', 'two', 'fish', '.'],
               ['red', 'fish', 'blue', 'fish', '.']]
        vocab = {'one': 3, 'fish': 1, 'two': 4, '.': 2, 'red': 5, 'blue': 6}
        actual = utils.indexize_doc(doc, vocab)
        expected = np.array([[3,1,4,1,2],[5,1,6,1,2]])
        npt.assert_array_equal(actual, expected)

class TestGetDocAndSentLength(unittest.TestCase):
    
    def setUp(self):
	    self.docs = pd.Series(
	    	[[[1]],
	    	 [[1,2],[1,2,3,4,5]],
	    	 [[1,2,3],[1,2,3,4,5]],
	    	 [[1],[1],[1,2,3]],
	    	 [[1,2],[1,2,3],[1,2,3,4],[1,2,3,4]]])

    def test_get_doc_and_sent_length(self):
    	actual = utils.get_doc_and_sent_length(self.docs)
    	expected = (4, 5)
    	self.assertEqual(actual, expected)

    def test_lower_doc_pct(self):
    	actual = utils.get_doc_and_sent_length(self.docs, doc_pct=0.5)
    	expected = (2, 5)
    	self.assertEqual(actual, expected)

    def test_lower_sent_pct(self):
    	actual = utils.get_doc_and_sent_length(self.docs, sent_pct=0.5)
    	expected = (4, 3)
    	self.assertEqual(actual, expected)

class TestDocsToTensor(unittest.TestCase):
    
    def setUp(self):
        self.vocab = {'the': 1, 'quick': 2, 'brown': 3, 'fox': 4}
        self.doc_len = 3
        self.sent_len = 4

    def test_long_sentence(self):
        docs = [[['the', 'quick', 'quick', 'brown', 'brown', 'fox'],
                 ['the', 'quick', 'quick', 'brown', 'fox'],
                 ['the', 'quick', 'quick', 'brown', 'fox']]]
        expected = np.array(
            [[[2, 3, 3, 4],
              [2, 2, 3, 4],
              [2, 2, 3, 4]]]
        )
        actual = utils.docs_to_tensor(docs,
                                      self.vocab,
                                      self.doc_len,
                                      self.sent_len)
        npt.assert_array_equal(actual, expected)

    def test_short_sentence(self):
        docs = [[['the', 'brown', 'fox'],
                 ['the', 'brown', 'fox'],
                 ['the', 'fox']]]
        expected = np.array(
            [[[1, 3, 4, 0],
              [1, 3, 4, 0],
              [1, 4, 0, 0]]]
        )
        actual = utils.docs_to_tensor(docs,
                                      self.vocab,
                                      self.doc_len,
                                      self.sent_len)
        npt.assert_array_equal(actual, expected)

    def test_long_doc(self):
        docs = [[['foo', 'bar'],
                 ['the', 'quick', 'brown', 'fox'],
                 ['the', 'quick', 'brown', 'fox'],
                 ['the', 'quick', 'brown', 'fox']]]
        expected = np.array(
            [[[1, 2, 3, 4],
              [1, 2, 3, 4],
              [1, 2, 3, 4]]]
        )
        actual = utils.docs_to_tensor(docs,
                                      self.vocab,
                                      self.doc_len,
                                      self.sent_len)
        npt.assert_array_equal(actual, expected)

    def test_short_doc(self):
        docs = [[['the', 'quick', 'brown', 'fox'],
                 ['the', 'quick', 'brown', 'fox']]]
        expected = np.array(
            [[[1, 2, 3, 4],
              [1, 2, 3, 4],
              [0, 0, 0, 0]]]
        )
        actual = utils.docs_to_tensor(docs,
                                      self.vocab,
                                      self.doc_len,
                                      self.sent_len)
        npt.assert_array_equal(actual, expected)

    def test_multiple_docs(self):
        docs = [[['the'],
                 ['the', 'quick', 'quick', 'brown', 'fox']],
                [['the', 'quick', 'brown', 'fox'],
                 ['the', 'quick', 'brown', 'fox'],
                 ['the', 'quick', 'brown', 'fox'],
                 ['the', 'quick', 'brown', 'fox']]]
        expected = [[[1,0,0,0],
                     [2,2,3,4],
                     [0,0,0,0]],
                    [[1,2,3,4],
                     [1,2,3,4],
                     [1,2,3,4]]]
        actual = utils.docs_to_tensor(docs,
                                      self.vocab,
                                      self.doc_len,
                                      self.sent_len)
        npt.assert_array_equal(actual, expected)