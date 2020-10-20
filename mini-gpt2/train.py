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

from tb import TransformerBlock
from tpe import TokenAndPositionEmbedding
from vocab import build_vocab

GPU = True
EPOCHS = 30

vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

ds_name = "bookset"
start_prompt = b"lui non avrebbe mai creduto che"
directories = [
    '../' + ds_name,
]

BEST_MODEL_FILE = ds_name + "_best_model.hdf5"


if not GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    

def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen=maxlen, vocab_size=vocab_size, embed_dim=embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam", loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model


def load_model():
    model = keras.models.load_model(BEST_MODEL_FILE, custom_objects={
        'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
        'TransformerBlock': TransformerBlock
    })
    return model

batch_size = 32

# Create a list all files
text_ds, index_to_word, word_to_index = build_vocab(directories, batch_size, vocab_size, maxlen)


def tokenizeString(s):
    return [word_to_index.get(_, 1) for _ in s.split()]

    

class TextGenerator(keras.callbacks.Callback):
    def __init__(self, max_tokens, start_tokens):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.k = 10

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = b" ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")



try:
    model = load_model()
    print ("Loaded saved model")
except:
    model = create_model()

model.summary()


text_gen_callback = TextGenerator(40, tokenizeString(start_prompt))
checkpoint = ModelCheckpoint(BEST_MODEL_FILE, monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

model.fit(text_ds, verbose=1, epochs=EPOCHS, callbacks=[text_gen_callback, checkpoint])

model.summary()