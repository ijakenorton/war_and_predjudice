from transformer import masked_loss
import keras
import numpy as np


class predictTextDataGenerator(keras.utils.Sequence):

    def __init__(self, ids, seq_len, batch_size):
        """
        Constructor for the data generator.

        param ids: A list of integers representing the tokens in the training text.
        param seq_len: The length of the input and target sequences for the transformer model.
        param batch_size: The number of sequences in each batch for training
        """

        # Save all the training text and parameters of the data generator
        self.ids = ids
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Compute the number of samples - it's the length of the text minus the sequence length
        self.num_samples = len(self.ids) - seq_len - 1
        # Run the on_epoch_end() method - which scrambles the data into the batchs
        # (this method will also be run during trainin at the end of each training epoch)
        self.on_epoch_end()

    def __len__(self):
        """
        You must provide this method to tell the model how many batches there are in an epoch.

        returns The number of batches in an epoch.
        """
        return self.num_samples // self.batch_size

    def __data_generation(self, list_IDs_temp):
        """
        This method generates a batch of training data for the model.  It's called by the
        __getitem__() method which is called by the model during training.

        param list_IDs_temp: A list of integers representing the indexes of the training data
        to be included in the batch.
        returns A tuple of input and target output sequences for the model.
        """

        # The input and target sequences are both of shape (batch_size, seq_len) and
        # are integer ids of the tokens (the transformer model will convert these to word vectors based
        # on the embedding you specify)
        X = np.zeros((self.batch_size, self.seq_len), dtype="int")
        y = np.zeros((self.batch_size, self.seq_len), dtype="int")

        # For each index in the list of indexes...
        for i, ID in enumerate(list_IDs_temp):
            # ...get the sequence of tokens from the training of length seq_len starting at
            # index ID.  In this case the input sequence is the sequence spans the entire
            # length of seq_len, but you might also train on shorter sequences, padded with zeros.
            # makse_loss will included padded inputs/outputs.
            X[i, : self.seq_len] = self.ids[ID : ID + self.seq_len]
            # ....and the self.sequence of target tokens, which is the sequence of tokens from the
            # training text of length self.seq_len starting at index ID+1 (offset by one, to match
            # the next word in the output to current word in the input)
            y[i, : self.seq_len] = self.ids[ID + 1 : ID + self.seq_len + 1]

        return X, y

    def __getitem__(self, index):
        """
        This method is called by the model during training to get a batch of training data.

        param index: The index of the batch to get.
        returns A tuple of input and target output sequences for the model.
        """

        # Generate indexes of the batch
        list_IDs_temp = self.list_IDs[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        This method is called at the end of each epoch of training.  It shuffles the data
        so that the batches are different in each epoch.
        """

        # Shuffle the tokens
        self.list_IDs = np.arange(self.num_samples)
        np.random.shuffle(self.list_IDs)
