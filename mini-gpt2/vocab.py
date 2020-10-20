#https://keras.io/examples/generative/text_generation_with_miniature_gpt/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import re
import string
import random


class Vocabulary:
    vectorize_layer = None 

    def getDataset(self):
        pass


vectorize_layer = None 

def prepare_lm_inputs_labels(text):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y



def custom_standardization(input_string):
    """ Remove html line-break tags and handle punctuation """
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


def build_vocab (directories, batch_size, vocab_size, maxlen):
    global vectorize_layer

    # Create a list all files
    filenames = []
    for dir in directories:
        for f in os.listdir(dir):
            filenames.append(os.path.join(dir, f))

    print(f"{len(filenames)} files")

    # Create dataset from text files
    random.shuffle(filenames)
    text_ds = tf.data.TextLineDataset(filenames)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)

    # Create vectcorization layer and adapt it to the text
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size - 1,
        output_mode="int",
        output_sequence_length=maxlen + 1,
    )
    vectorize_layer.adapt(text_ds)
    vocab = vectorize_layer.get_vocabulary()

    word_to_index = {}
    for index, word in enumerate(vocab):
        word_to_index[word] = index

    text_ds = text_ds.map(prepare_lm_inputs_labels)
    text_ds = text_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return (text_ds, vocab, word_to_index)


