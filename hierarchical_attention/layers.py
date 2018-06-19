import numpy as np
from keras import backend as K
from keras.regularizers import Regularizer
from keras.layers import  Dot, TimeDistributed, Layer
from keras.engine.topology import _object_list_uid
from keras.utils.generic_utils import has_arg

class AttentionLayer(Layer):
    r"""
    Keras Layer for a attention mechanism [1]_.

    The attention layer provides a mechanism to de-emphasize non-topical items
    in a sequence during training.

    Trainable Weights
    ----------
    W : keras tensor of shape (`word_embed_dim`, `word_embed_dim`)
        If `None`, `W` will be randomly initialized.
    b : keras tensor of shape (`embed_dim`,), optional
        If `None`, `b` will be the zero vector.
    u : keras tensor of shape (`embed_dim`,), optional
        If `None`, `u` will be the randomly initialized.

    Notes
    -----
    The paper implements the attention through a bilinear softmax
        1.    :math:`u_i = \tanh(Wh_i + b)`
        2.    :math:`\alpha_i = softmax(u_i^Tu)`

    where the :math:`h_i` are contextualized embeddings (i.e. output from a bi-
    RNN) of the words in the sentence or sentences in the document.

    The idea of the attention layer is to de-emphasize irrelevant words (or
    sentences) during the training process.

    .. [1] Yang, Z.; Yang, D.; Dyer; He; Smola; Hovy. "Hierarchical Attention
    Networks for Document Classification", http://aclweb.org/anthology/N/N16/N16-1174.pdf
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        """
        Create trainable weights for the attention matrix `W`, the bias vector `b`,
        and the "context vector" `u`.
        See equations (5) and (8) page 1482 from above-referenced HAN paper.

        Parameters
        ----------
        input_shape : (tuple)
            Dims of the input tensor.   This is expected to be a triple with 
            entries (`batch_size`, `sentlen`, `hidden_dim`)
        """
        hidden_dim = input_shape[2]
        self.W = self.add_weight(
            name='{}_W'.format(self.name),
            shape=(hidden_dim, hidden_dim,),
            initializer='uniform',
            trainable=True)
        self.b = self.add_weight(
            name='{}_b'.format(self.name),
            shape=(hidden_dim,),
            initializer='zeros',
            trainable=True)
        self.u = self.add_weight(
            name='{}_u'.format(self.name),
            shape=(hidden_dim,),
            initializer='uniform',
            trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, mask=None):
        return None

    def call(self, x, mask=None):
        r"""
        Provide attention coefficients (equations 6 and 9 from HAN) for given
        sentence.

        Parameters
        ----------
        x : Keras tensor of shape (`batch_size`, `seq_len`, `hidden_dim`)

        mask : boolean keras tensor of shape (`batch_size`, `seq_len`), optional
            If `None`, compute without masking. If not `None` then any masked
            word vector in `x` will, we apply a re-weighted softmax, such that
            the attention weights for masked tokens are 0. See Notes below.

        Returns
        -------
        Keras tensor of shape (`batch_size`, `seq_len`)

        Notes
        -----

        Strictly speaking, in calculating the quantity `u_i`, we should take the 
        transpose of `W`, since we are taking the transpose of `X` and left-
        multiplying. However, `W` is an arbitrary square matrix and it exists
        only for the purposes of the present calculation.  So whether we
        initialize `W` or its transpose doesn't matter.

        When padding sequences with a padding tokens, the padding tokens may be
        masked so that their attention weights are always 0 and they do not con-
        tribute to any the updates of any trainable weights in the attention layer.

        A Keras masking layer--equivalently, setting the optional `mask_zero` arg
        of an Embedding layer to `True`--will create a boolean mask with value
        `False` if the value of the input is equal to the mask value.

        .. [1] Yang, Z.; Yang, D.; Dyer; He; Smola; Hovy. "Hierarchical Attention
        Networks for Document Classification", http://aclweb.org/anthology/N/N16/N16-1174.pdf
        """
        u_i = K.tanh(K.dot(x, self.W) + self.b)
        v = K.squeeze(K.dot(u_i, K.expand_dims(self.u)), axis=-1)
        attention = softmax_masked(v, mask)

        return attention
        
    def compute_output_shape(self, input_shape):
        """Compute output shape of layer"""
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        return (batch_size, sequence_length)

def softmax_masked(x, mask=None):
    """
    Softmax computed with an optional mask

    Parameters
    ----------
    x : keras tensor rank 2, shape is (`batch_size`, `seq_len`)

    mask : boolean keras tensor of rank 2, shape (`batch_size`, `seq_len`), optional

    Returns
    -------
    sm : keras tensor rank 2, shape is (`batch_size`, `seq_len`)
    """
    if mask is None:
        sm = K.softmax(x)
    else:
       casted_mask = K.cast(mask, dtype=x.dtype)
       # subtract min first so that, after masking, the masked elements are the smallest
       z = (x - K.min(x, axis=1, keepdims=True)) * casted_mask
       # Now subtract max of non-masked elements, and 0 for masked, in order
       # to prevent exponentiating a very large number
       e = K.exp(z - K.max(z, axis=1, keepdims=True)) * casted_mask
       s = K.sum(e, axis=1, keepdims=True) + K.epsilon()
       sm = e/s
    return sm

class TimeDistributedWithMasking(TimeDistributed):
    """
    Subclass of keras.layers.wrappers.TimeDistributed that implements masking.

    This implementation is specific to the usage in HAN, and is not necessarily
    intended to be generally applicable
    """

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def call(self, inputs, training=None, mask=None):
        # Copied from https://github.com/fchollet/keras/blob/master/keras/layers/wrappers.py#L100
        # (tag 2.1.0) with our modifications marked in comments
        kwargs = {}
        if has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        # Our modification
        if has_arg(self.layer.call, 'mask'):
            kwargs['mask'] = mask
        # end our modification
        uses_learning_phase = False

        input_shape = K.int_shape(inputs)

        input_length = input_shape[1]
        if not input_length:
            input_length = K.shape(inputs)[1]
        # Shape: (num_samples * timesteps, ...). And track the
        # transformation in self._input_map.
        input_uid = _object_list_uid(inputs)
        inputs = K.reshape(inputs, (-1,) + input_shape[2:])
        self._input_map[input_uid] = inputs
        # (num_samples * timesteps, ...)
        # Our modification
        if kwargs.get('mask', None) is not None:
            mask_shape = K.int_shape(mask)
            kwargs['mask'] = K.reshape(kwargs['mask'], (-1,) + mask_shape[2:])
        # end our modification
        y = self.layer.call(inputs, **kwargs)
        if hasattr(y, '_uses_learning_phase'):
            uses_learning_phase = y._uses_learning_phase
        # Shape: (num_samples, timesteps, ...)
        output_shape = self.compute_output_shape(input_shape)
        y = K.reshape(y, (-1, input_length) + output_shape[2:])

        # Apply activity regularizer if any:
        if (hasattr(self.layer, 'activity_regularizer') and
           self.layer.activity_regularizer is not None):
            regularization_loss = self.layer.activity_regularizer(y)
            self.add_loss(regularization_loss, inputs)

        if uses_learning_phase:
            y._uses_learning_phase = True
        return y

class WeightedAverage(Layer):
    """
    Layer that computes the weighted average of a sequence of vectors. It also
    implements masking.

    For example, when using an attention mechanism, the weights would be the
    attention weights, and the vectors would be the outputs at each time step
    from a recurrent neural network.

    When called, expects an input list of length 2: `[weights, vectors]`. 
    The order of `weights` should be 1 less than the order of `vectors`, and the
    shape of `weights` should match the shape of `vectors` up to the last 
    dimension of `vectors`.

    Examples
    --------

    For a single sequence:
    `weights` should have dimension (`batch_size`, `seq_len`).
    `vectors` should have dimension (`batch_size`, `seq_len`, `embed_dim`).
    The computed weighted average tensor will have dimension 
    (`batch_size`,`embed_dim`)

    For a sequence of sequences:
    `weights` should have dimension (`batch_size`, `outer_seq_len`, `inner_seq_len`).
    `vectors` should have dimension (`batch_size`, `outer_seq_len`, `inner_seq_len`, `embed_dim`).
    The computed weighted average tensor will have dimension 
    (`batch_size`, `outer_seq_len`, `embed_dim`).
    """
    def __init__(self, **kwargs):
        super(WeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `WeightedAverage` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return

        self._check_shape(shape1, shape2)

    def call(self, inputs, mask=None):
        """
        Compute the weighted average of a collection of vectors with a set of weights.

        Arguments
        ---------
        inputs : a list of keras tensors with shapes (`batch_size`, `seq1_len`,
            ..., `seqn_len`) and (`batch_size`, `seq1_len`, ..., `seqn_len`,
            `embed_dim`), respectively.
        mask : ignored

        Returns
        -------
        wtd_avg : a keras tensor with shape (`batch_size`, `seq1_len`, ...,
            `seq(n-1)_len`, `embed_dim`)
        """
        weights = K.expand_dims(inputs[0])
        vectors = inputs[1]
        wtd_vectors = weights * vectors
        wtd_avg = K.sum(wtd_vectors, axis=-2)
        return wtd_avg

    def compute_mask(self, inputs, mask=None):
        """
        Then the output-mask value is `True` as long any of the input vectors
        have a `True` mask value. Otherwise, it is `False`.
        """
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if mask[0] is not None:
            raise ValueError('Attention mask should be None.')
        if mask[1] is None:
            return None
        return K.any(mask[1], axis=-1)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `WeightedAverage` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        self._check_shape(shape1, shape2)

        output_shape = shape1[:-1] + [shape2[-1]]
        # shape1.pop(self.axes)
        # shape2.pop(self.axes)
        # [shape2.pop(0) for _ in range(self.axes)]
        # output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)

    def _check_shape(self, shape1, shape2):
        if len(shape1) != len(shape2) - 1:
            raise ValueError('The order of the `weights` tensor should be 1 '
                             'less than the order of the `vectors` tensor.')
        
        if shape1 != shape2[:-1]:
            raise ValueError('Incompatible input shapes.')