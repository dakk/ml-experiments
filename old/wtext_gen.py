# https://www.tensorflow.org/tutorials/text/text_generation

# In questo script usiamo le parole invece che le lettere; la RNN quindi stabilisce quale e' la parola con maggiore probabilita' di comparire

import tensorflow as tf

import numpy as np
import os
import time
import re


TRAIN = False
GPU = False
EPOCHS = 50
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

# The maximum length sentence you want for a single input in characters
seq_length = 64


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints_tg2'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


if not GPU:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)



def format_text (text):
    # text = re.sub(r'[^\w]', ' ', text)
    return text.replace('\n', ' ').replace('...', ' ... ').replace('\r', ' ').replace('[', '').replace(']', '').replace('"', ' " ').replace(':', ' : ').replace('\'', ' \' ').replace('(', ' ( ').replace(')', ' ) ').replace('-', ' - ').replace('.', ' . ').replace(';', ' ; ').replace(',', ' , ').replace('!', ' ! ').replace('?', ' ? ').lower().split(' ')


#path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
path_to_file = 'guerraepace.txt'

# Read, then decode for py2 compat.
text = format_text(open(path_to_file, 'rb').read().decode(encoding='utf-8'))
# length of text is the number of words in it
print('Length of text: {} words'.format(len(text)))


vocab = sorted(set(text))
vocab_size = len(vocab)
print('{} unique words'.format(vocab_size))

# print (vocab)


# Creating a mapping from unique words to indices
word2idx = {u:i for i, u in enumerate(vocab)}
idx2word = np.array(vocab)

text_as_int = np.array([word2idx[c] for c in text])

# print('{')
# for char,_ in zip(word2idx, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), word2idx[char]))
# print('  ...\n}')

# # Show how the first 13 characters from the text are mapped to integers
# print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))


examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# for i in char_dataset.take(5):
#     print(idx2word[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# for item in sequences.take(5):
#     print(repr(' '.join(idx2word[item.numpy()])))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# for input_example, target_example in  dataset.take(1):
#     print('Input data: ', repr(''.join(idx2word[input_example.numpy()])))
#     print('Target data:', repr(''.join(idx2word[target_example.numpy()])))


# for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
#     print("Step {:4d}".format(i))
#     print("  input: {} ({:s})".format(input_idx, repr(idx2word[input_idx])))
#     print("  expected output: {} ({:s})".format(target_idx, repr(idx2word[target_idx])))



print ("Shuffling")
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)



# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
    


print ("Building model")

if TRAIN:
    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.summary()


    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

    print("Input: \n", repr(" ".join(idx2word[x] for x in input_example_batch[0])))
    print()
    print("Next Char Predictions: \n", repr(" ".join(idx2word[sampled_indices ])))


    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

    model.compile(optimizer='adam', loss=loss)



    try:
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    except:
        print ("No checkpoint, starting from first epoch")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)


    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    tf.train.latest_checkpoint(checkpoint_dir)








model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()


def generate_text(model, start_string):
    start_arr = format_text(start_string)

    # Number of word to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [word2idx[s] for s in start_arr]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2word[predicted_id])

    return (start_string + ' ' + ' '.join(text_generated).replace(' . . .', '...').replace('\r', ' ').replace(' "', '"').replace(' :', ':').replace(' \'', '\'').replace(' (', '(').replace(' )', ')').replace(' .', '.').replace(' ;', ';').replace(' ,', ',').replace(' !', '!').replace(' ?', '?'))


print(generate_text(model, start_string=u"durante la battaglia, pierre ebbe un idea"))

