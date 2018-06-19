'''
This example script creates a synthetic data set designed to mimic some features
we see in natural language (e.g. negation, long-term dependence between words/
phrases), as well as to demonstrate that the network properly handles sentences
and documents of variable lengths (i.e. via padding and masking).
'''

import numpy as np
import random
import hierarchical_attention as han

np.random.seed(1004)
random.seed(198705)

def main():
    # Params
    sent_len = 9
    doc_len = 5
    vocab_size = 10
    gru_dim = 3
    word_embed_dim = 3
    n_classes = 1
    n_samples = 120000
    validation_split = 0.2
    epochs = 6

    # Synthetic data
    x, y, x_test, y_test = _create_synthetic_data(n_samples,
                                                  doc_len,
                                                  sent_len,
                                                  vocab_size,
                                                  validation_split)

    # Model
    model = han.YangEtAl2016(vocab_size=vocab_size,
                             sent_len=sent_len,
                             doc_len=doc_len,
                             word_embed_dim=word_embed_dim,
                             gru_dim=gru_dim,
                             n_classes=1)

    model.fit(x, y, epochs=epochs, validation_data=[x_test, y_test])

    model.han.summary()

    # Results
    print("inputs:\n", x[:5])
    print("actual:\n", y[:5])
    print("predicted:\n", np.round(model.han.predict(x[:5]), decimals=2))
    print("sentence attention:\n",
          np.round(model.sentence_attention.predict(x[:5]), decimals=2))
    print("word attentions\n",
          np.round(model.td_word_attention.predict(x[:5]), decimals=2))
    print("word vectors:\n",
          np.round(model.word_embedding_layer.get_weights(), decimals=3))


def _create_synthetic_data(n_samples, doc_len, sent_len, vocab_size, validation_split,):
    '''
    Create a synthetic dataset for training and testing the hierarchical 
    attention network from [1]_.

    Inputs are sequences of sequences of integers, represented as a numpy array
    with dimensions (`n_samples`, `doc_len`, `sent_len`).
    
    The labels are a complex function of the inputs.

    Arguments
    ---------
    n_samples: `int` data-set size.
    doc_len: `int` maximum document length (must be >= 2)
    sent_len: `int` maximum sentence length (must be >= 5)
    vocab_size: `int` number of possible unique tokens in the data (must be 10)
    validation_split: `float` fraction of data to randomly assign to test set

    Returns
    -------
    x: `np.array` of shape (`(1-validation_split) * n_samples`, `doc_len`, `sent_len`),
        consisting of training inputs.
    y: `np.array` of shape (`(1 - validation_split) * n_samples`,), consisting of
        training labels.
    x_test: `np.array` of shape (`validation_split * n_samples`, `doc_len`, `sent_len`),
        consisting of training inputs.
    y_test: `np.array` of shape (`validatoin_split * n_samples`,), consisting of 
        test labels.

    Notes
    -----
    All labels are initialized to 0.
    If a certain key phrase (e.g. [1,2,3]) appears in a specific sentence (e.g. 
    2nd or 5th) in the input document _AND_ a long-term token (e.g. 5) also app-
    ears at the end of the same sentence, then the label is flipped to 1. This
    simulates a long-term dependency between the key phrase and the long-term
    token. However, if the long-term token is preceded by a negation token (e.g.
    9), then the label is flipped back to 0. This simulates negation.

    .. [1] Yang, Z.; Yang, D.; Dyer; He; Smola; Hovy. "Hierarchical Attention
    Networks for Document Classification", http://aclweb.org/anthology/N/N16/N16-1174.pdf
    '''

    assert validation_split > 0.
    assert validation_split < 1.
    assert vocab_size == 10
    assert n_samples > 0
    assert doc_len >= 2
    assert sent_len >= 5

    key_phrase = [1,2,3]
    long_term_token = 5
    negation_token = 9

    # Initialize input documents as random tokens
    # (i.e. integers in range(vocab_size))
    x = np.random.choice(vocab_size, (n_samples, doc_len, sent_len))
    # Randomly set some sentences to 0 to simulate variable doc lengths
    for i in np.random.choice(n_samples, int(.4*n_samples), replace=False):
        x[i, random.choice(range(1,6)):,:] = 0

    # Initialize all labels to 0.
    y = np.zeros(n_samples)
    
    # Randomly insert a key-phrase into the first half of either the 2nd or 4th sentence.
    # If the long-term token is also at the end of the 2nd or 5th sentence, then set the label to 1.
    # However, if the negation token precedes the long-term token, set the label back to 0.
    for i in np.random.choice(n_samples, int(.75*n_samples), replace=False):
        s = random.choice([1,doc_len - 1])
        w = random.choice(range(sent_len - 5))
        w2 = random.choice([sent_len - 1, sent_len - 2])
        # Key-phrase
        x[i, s, w:(w+len(key_phrase))] = key_phrase
        if random.random() > 0.2:
            # Long-term dependence
            x[i, s, w2] = long_term_token
            y[i] = 1
            if random.random() > 0.6:
                # Negation
                x[i, s, w2-1] = negation_token
                y[i] = 0

    test_indices = np.random.choice(n_samples,
                                    int(validation_split*n_samples),
                                    replace=False)
    test_mask = np.zeros(n_samples, dtype=np.bool)
    test_mask[test_indices] = True
    x_train = x[np.logical_not(test_mask)]
    y_train = y[np.logical_not(test_mask)]
    x_test = x[test_mask]
    y_test = y[test_mask]

    return (x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()