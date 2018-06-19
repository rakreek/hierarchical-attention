import numpy as np
import pandas as pd
import hierarchical_attention.batching as hanb

import unittest
from numpy.testing import assert_array_equal

class TestBatchGenerator(unittest.TestCase):

    def setUp(self):

# These tests currently require that `docs` (below) remain in its current order. 
# If `docs` is reordered or expanded, tests might start failing (this is a result of 
# sort_values in ../hierarchical_attention/batching.py using the quicksort algorithm). 

        docs = [
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
            [[3, 3], [3, 3]],
            [[4, 4], [4, 4]],
            [[5, 5], [5, 5]],
            [[6, 6], [6, 6], [6, 6]],
            [[7, 7], [7, 7], [7, 7]],
            [[8, 8], [8, 8], [8, 8]],
            [[9, 9], [9, 9], [9, 9], [9, 9]],
            [[10, 10]]
        ]

        self.text_col = 'Doc'
        self.label_col = 'Label'
        self.key_col = 'Key'
        self.padding_idx = -1
        self.df = pd.DataFrame({self.text_col:docs})
        self.df[self.label_col] = self.df.index + 1
        self.df['doc_len'] = self.df[self.text_col].apply(len)
        self.grouped = self.df.groupby(by='doc_len')

        self.batch_gen = hanb.BatchGenerator(
            text_col=self.text_col,
            label_col=self.label_col,
            key_col=self.key_col,
            batch_size=3,
            padding_idx=self.padding_idx,
            shuffle=False)
        
    def _get_results_for_n_epochs(self, num_epochs, batch_size, *args, **kwargs):
         self.batch_gen.batch_size = batch_size

         num_batches_per_epoch = self.batch_gen.num_batches_per_epoch(self.df, **kwargs)

         trainer = self.batch_gen.fit_generator(
             df=self.df[[self.text_col, self.label_col]],
             **kwargs)

         batches = []

         while len(batches) < num_epochs * num_batches_per_epoch:
             batches.append(next(trainer))

         return batches

    def test_labels_in_one_epoch_batch_3_no_shuffle(self):

        self.assertEqual(
            [[1, 2, 3], [6, 7, 8], [10, 4, 5], [9]],
            [batch[1] for batch in self.batch_gen._generate_batches_for_one_epoch(self.grouped, False, 1)])

        self.assertEqual(
            [[1, 2, 3], [6, 7, 8], [10, 4, 5]],
            [batch[1] for batch in self.batch_gen._generate_batches_for_one_epoch(self.grouped, False, 2)])

        self.assertEqual(
            [[10], [1, 2, 3], [4, 5], [6, 7, 8], [9]],
            [batch[1] for batch in self.batch_gen._generate_batches_for_one_epoch(self.grouped, True, 1)])

        self.assertEqual(
            [[1, 2, 3], [4, 5],  [6, 7, 8], [10, 9]],
            [batch[1] for batch in self.batch_gen._generate_batches_for_one_epoch(self.grouped, True, 2)])

    def test_output_evenly_divisible(self):
        self.batch_gen.batch_size = 5
        
        expected_output = [
            np.array([
                [[1, 1], [1, 1]],
                [[2, 2], [2, 2]],
                [[3, 3], [3, 3]],
                [[4, 4], [4, 4]],
                [[5, 5], [5, 5]]]),
            np.array([
                [[10, 10], [-1, -1], [-1, -1], [-1, -1]],
                [[6, 6], [6, 6], [6, 6], [-1, -1]],
                [[7, 7], [7, 7], [7, 7], [-1, -1]],
                [[8, 8], [8, 8], [8, 8], [-1, -1]],
                [[9, 9], [9, 9], [9, 9], [9, 9]]])
        ]
        
        actual = [
            batch[0] for batch in self.batch_gen._generate_batches_for_one_epoch(self.grouped, False, 2)
        ]
        
        for e, a in zip(expected_output, actual):
            assert_array_equal(e, a)
        
    def test_num_batches_per_epoch_batch_3(self):
        self.batch_gen.shuffle = True 

        # labels: [1, 2, 3], [6, 7, 8], [10, 4, 5], [9]
        self.assertEqual(
            4,
            self.batch_gen.num_batches_per_epoch(self.df, allow_diff_batch_size=False, min_batch_size=1))

        # by default won't emit final batch of size 1
        # labels: [1, 2, 3], [6, 7, 8], [10, 4, 5]
        self.assertEqual(
            3,
            self.batch_gen.num_batches_per_epoch(self.df, allow_diff_batch_size=False))

        # labels: [1, 2, 3], [4, 5],  [6, 7, 8], [9], [10]
        self.assertEqual(
            5,
            self.batch_gen.num_batches_per_epoch(self.df, allow_diff_batch_size=True, min_batch_size=1))

        # labels: [1, 2, 3], [4, 5],  [6, 7, 8], [9, 10]
        self.assertEqual(
            4,
            self.batch_gen.num_batches_per_epoch(self.df, allow_diff_batch_size=True, min_batch_size=2))
        
    def test_labels_in_one_epoch_batch_4_no_shuffle(self):

        self.batch_gen.batch_size = 4 

        self.assertEqual(
            [[1, 2, 3, 4], [10, 5, 6, 7], [8, 9]],
            [batch[1] for batch in self.batch_gen._generate_batches_for_one_epoch(self.grouped, False, 1)])

        self.assertEqual(
            [[1, 2, 3, 4], [10, 5, 6, 7], [8, 9]],
            [batch[1] for batch in self.batch_gen._generate_batches_for_one_epoch(self.grouped, False, 2)])

        self.assertEqual(
            [[10], [1, 2, 3, 4], [5], [6, 7, 8], [9]],
            [batch[1] for batch in self.batch_gen._generate_batches_for_one_epoch(self.grouped, True, 1)])

        self.assertEqual(
            [[1, 2, 3, 4], [6, 7, 8], [10, 5, 9]],
            [batch[1] for batch in self.batch_gen._generate_batches_for_one_epoch(self.grouped, True, 2)])

    def test_fit_gen_multiple_epochs_labels(self):

        batch_size = 3

        for num_epochs in [1, 2, 5]:
            self.assertEqual(
                [[1, 2, 3], [6, 7, 8], [10, 4, 5], [9]]*num_epochs,
                [batch[1] for batch in self._get_results_for_n_epochs(num_epochs, batch_size, allow_diff_batch_size=False, min_batch_size=1)])

    def test_fit_gen_multiple_epochs_outputs(self):

        batch_size = 3
        expected_one_epoch = [
            np.array([
                [[1, 1], [1, 1]],
                [[2, 2], [2, 2]],
                [[3, 3], [3, 3]]]),
            np.array([
                [[6, 6], [6, 6], [6, 6]],
                [[7, 7], [7, 7], [7, 7]],
                [[8, 8], [8, 8], [8, 8]]]),
            np.array([
                [[10, 10], [-1, -1]],
                [[4, 4], [4, 4]],
                [[5, 5], [5, 5]]]),
            np.array([
                [[9, 9], [9, 9], [9, 9], [9, 9]]])
        ]

        for num_epochs in [1, 2, 5]:
            actual = [
                batch[0] for batch in self._get_results_for_n_epochs(
                    num_epochs, batch_size, allow_diff_batch_size=False, min_batch_size=1)
            ]
            expected = expected_one_epoch*num_epochs
            for e, a in zip(expected, actual):
                assert_array_equal(e, a)

    def test_predict_generator(self):

        batch_size = 3

        expected = [
            np.array([
                [[1, 1], [1, 1]],
                [[2, 2], [2, 2]],
                [[3, 3], [3, 3]]]),
            np.array([
                [[6, 6], [6, 6], [6, 6]],
                [[7, 7], [7, 7], [7, 7]],
                [[8, 8], [8, 8], [8, 8]]]),
            np.array([
                [[10, 10], [-1, -1]],
                [[4, 4], [4, 4]],
                [[5, 5], [5, 5]]]),
            np.array([
                [[9, 9], [9, 9], [9, 9], [9, 9]]])
        ]

        self.df[self.key_col] = ['a1', 'b2', 'c3', 'd4', 'e5', 'f6', 'g7', 'h8', 'i9', 'j10']

        actual = self.batch_gen.predict_generator(
            df=self.df[[self.text_col, self.key_col]])

        for e, a in zip(expected, actual):
            assert_array_equal(e, a)

        with self.assertRaises(StopIteration):
            next(actual)

        expected = ['a1', 'b2', 'c3', 'f6', 'g7', 'h8', 'j10', 'd4', 'e5', 'i9']
        actual = self.batch_gen.ordered_indices
        self.assertEqual(
            expected,
            actual
        )
        
    def test_shuffle_reverts(self):
        self.batch_gen.shuffle = True
        self.batch_gen.predict_generator(df=self.df[self.text_col])
        actual = self.batch_gen.shuffle
        self.assertEqual(True, actual)
