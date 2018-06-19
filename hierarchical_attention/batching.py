import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import random

class BatchGenerator:
    """ A batch generator that batches together documents of the same length to accelerate training speed.
    
    Parameters
    ----------
    text_col : str
        Name of the dataframe column that contains the documents.
    label_col : str
        Name of the dataframe column that contains the labels.
    key_col : str
        Name of dataframe column containing unique identifier for each record.
        (Optional, if unspecified, the dataframe's original index is used.)
    batch_size : int
        Number of records in one batch.
    padding_idx : int
        Dictionary index of the padding token.
    shuffle: bool
        Flag for whether the order of the batches and records within each batch need to be shuffled.
        (Will automatically be set to `False` when generating test data (`predict_generator`).)

    Attributes
    ----------
    ordered_indices : list
        Final order of the data generated (in terms of each record's index)
    
    Methods
    -------
    fit_generator(self, df, allow_diff_batch_size=False, min_batch_size=2, multiple_epochs=True) : 
        Generate batches of padded training data.
    predict_generator(self, df, *args, **kwargs) : Generate batches of padded testing data.
    num_batches_per_epoch(self, *args, **kwargs) : Calculate the number of steps per epoch.
    """

    def __init__(self, text_col, label_col='label_col', key_col='key_col', batch_size=64, padding_idx=-1, shuffle=True):
        self.text_col = text_col
        self.label_col = label_col
        self.key_col = key_col
        self.batch_size = batch_size
        self.padding_idx = padding_idx
        self.shuffle = shuffle
        self.ordered_indices = []

    @property
    def columns(self):
        return [self.text_col, self.label_col]

    def fit_generator(self, df, allow_diff_batch_size=False, min_batch_size=2, multiple_epochs=True):
        '''
        Generate batches of padded training data. 

        Parameters
        ----------
        df : pandas dataframe
            Dataframe containing data to be batched.
        allow_diff_batch_size : bool
            Flag for whether generator can yield batches of different sizes in the middle of an epoch.
            (Generator can always yield a differently-sized final batch for an epoch.)
        min_batch_size : int
            Minimum number of examples to yield in a training batch.
            (Must always be >= 2 when using `fit_generator`.)
        multiple_epochs : bool
            Flag for whether to generate batches for multiple epochs.
            (Should be `True` when using `fit_generator`.) 
        
        Yields
        ------
        batch : tuple of types (numpy array, list)
            The first element in the tuple (numpy array) contains the documents for the batch with each document
            padded to equal sentence and document lengths, and the second element is a list of the corresponding labels. 
        '''
        df = df.copy()
        df['doc_len'] = df[self.text_col].apply(len)
        
        if self.key_col in df.columns:
            df.set_index(self.key_col, inplace=True)
            
        if self.label_col not in df.columns:
            df[self.label_col] = -1
            
        grouped = df.groupby(by='doc_len')

        for batch in self._generate_batches_for_one_epoch(grouped, allow_diff_batch_size, min_batch_size):
            yield batch
        while multiple_epochs:
            for batch in self._generate_batches_for_one_epoch(grouped, allow_diff_batch_size, min_batch_size):
                yield batch

    def predict_generator(self, df, *args, **kwargs):
        """
        Generate batches of padded testing data. 
        
        Parameters
        ----------
        See `fit_generator`.
        
        Yields
        ------
        batch : numpy array
            The array contains the documents for the batch with each document padded to equal sentence and document lengths. 
            
        """
        if kwargs is None:
            kwargs = {}
        kwargs['multiple_epochs'] = False
        kwargs['min_batch_size'] = 1
        
        orig_shuffle = self.shuffle
        self.shuffle = False
        
        for batch in self.fit_generator(df=df, *args, **kwargs):
            yield batch[0]
            
        self.shuffle = orig_shuffle

    def num_batches_per_epoch(self, *args, **kwargs):
        """
        Calculate the number of steps per epoch. 
        
        Parameters
        ----------
        See `fit_generator`.
        
        Returns 
        -------
        num : int
            Number of steps in 1 epoch.
        """
        kwargs['multiple_epochs'] = False
        num = 0
        for batch in self.fit_generator(*args, **kwargs):
            num += 1
        return num

    def _generate_batches_for_one_epoch(
            self, grouped, allow_diff_batch_size, min_batch_size):

        self.ordered_indices = []
        leftovers = pd.DataFrame(columns=self.columns)

        grouped_keys = list(grouped.indices.keys())
        if self.shuffle:
            random.shuffle(grouped_keys)  # random.shuffle modifies in place

        for key in grouped_keys:
            for batch in self._batches_for_fixed_doc_len(grouped.get_group(key)):
                if len(batch[1]) == self.batch_size or (allow_diff_batch_size and len(batch[1]) >= min_batch_size):
                    self.ordered_indices.extend(batch[2])
                    yield batch[0], batch[1]
                else:
                    leftovers = leftovers.append(
                        pd.DataFrame({
                            self.text_col: batch[0].tolist(),
                            self.label_col: batch[1]
                        },
                        index=batch[2]))

        if len(leftovers) == 0:
            return

        leftovers['doc_len'] = leftovers[self.text_col].apply(len)
        leftovers.sort_values(by='doc_len', inplace=True)

        for batch in self._batches_for_var_doc_len(leftovers):
            if len(batch[1]) >= min_batch_size:
                self.ordered_indices.extend(batch[2])
                yield batch[0], batch[1]

    def _batches_for_fixed_doc_len(self, data):

        if self.shuffle:
            data = data.sample(frac=1)

        last_idx = 0
        while last_idx < data.shape[0]:
            batch = data.iloc[last_idx:last_idx + self.batch_size]
            last_idx += self.batch_size
            yield np.array(batch[self.text_col].tolist()), batch[self.label_col].tolist(), batch.index.tolist()

    def _batches_for_var_doc_len(self, data):

        last_idx = 0
        while last_idx < data.shape[0]:
            batch = data.iloc[last_idx:last_idx + self.batch_size]
            longest_doc = len(batch.iloc[-1][self.text_col])
            padded_docs = pad_sequences(
                batch[self.text_col].tolist(), maxlen=longest_doc, padding='post',
                truncating='post', value=self.padding_idx)
            last_idx += self.batch_size
            yield padded_docs, batch[self.label_col].tolist(), batch.index.tolist()




