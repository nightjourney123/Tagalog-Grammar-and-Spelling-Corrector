import nltk.translate.bleu_score as bleu
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm
import tensorflow as tf
import keras
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import layers
from keras import Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

tf.config.set_visible_devices([], 'GPU')

df = pd.read_csv('C:/Users/Xavier/Desktop/Text to FSL/processed_tagalog.csv')
df.columns = ["enc_input", "dec_input", "y"]
df["dec_output"] = df.dec_input
print(df)


## <start> AS BEGINNING TOKEN
## <end>  AS END TOKEN

df["dec_input"] = "<start> " + df["dec_input"]
df["dec_output"] = df["dec_output"] + " <end>"
print(df)


df_sampled = pd.concat((df[df.y == 1].sample(frac=0.99, random_state=1), df[df.y == 2]))
df_train, df_val = train_test_split(df_sampled, test_size=0.2, random_state=3, stratify=df_sampled.y)
df_train["dec_input"].iloc[0] = df_train.iloc[0]["dec_input"] + " <end>"
df_train[["enc_input", "dec_output", "y"]].to_csv("C:/Users/Xavier/Desktop/Text to FSL/training_091423.csv", index=False)
print("This is dataset for training : \n", df_train)


df_val[["enc_input", "dec_output", "y"]].to_csv("C:/Users/Xavier/Desktop/Text to FSL/validation_091423.csv", index=False)
print("This is dataset for validation : \n", df_val)

np.random.seed(5)
df_test = df.loc[
    np.random.choice(np.array([x for x in df.index.values if x not in df_sampled.index.values]), 1150, replace=False, )]
df_test[["enc_input", "dec_output", "y"]].to_csv("C:/Users/Xavier/Desktop/Text to FSL/testing_091423.csv", index=False)
print("This is dataset for testing : \n", df_test)


tk_inp = Tokenizer()
tk_inp.fit_on_texts(df_train.enc_input.apply(str))
print(len(tk_inp.word_index))


tk_out = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tk_out.fit_on_texts(df_train.dec_input.apply(str))
print(len(tk_out.word_index))

pickle.dump(tk_inp, open("tagalog_in_tokenizer_091423", "wb"))
pickle.dump(tk_out, open("tagalog_out_tokenizer_091423", "wb"))


class Dataset:
    def __init__(self, data, tk_inp, tk_out, max_len):
        self.encoder_inp = data["enc_input"].apply(str).values
        self.decoder_inp = data["dec_input"].apply(str).values
        self.decoder_out = data["dec_output"].apply(str).values
        self.tk_inp = tk_inp
        self.tk_out = tk_out
        self.max_len = max_len

    def __getitem__(self, i):
        self.encoder_seq = self.tk_inp.texts_to_sequences([self.encoder_inp[i]])
        self.decoder_inp_seq = self.tk_out.texts_to_sequences([self.decoder_inp[i]])
        self.decoder_out_seq = self.tk_out.texts_to_sequences([self.decoder_out[i]])
        self.encoder_seq = pad_sequences(self.encoder_seq, padding="post", maxlen=self.max_len)
        self.decoder_inp_seq = pad_sequences(self.decoder_inp_seq, padding="post", maxlen=self.max_len)
        self.decoder_out_seq = pad_sequences(self.decoder_out_seq, padding="post", maxlen=self.max_len)
        return self.encoder_seq, self.decoder_inp_seq, self.decoder_out_seq

    def __len__(self):
        return len(self.encoder_inp)


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, batch_size, dataset):
        self.dataset = dataset
        self.batch_size = batch_size
        self.totl_points = self.dataset.encoder_inp.shape[0]

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        batch_enc = []
        batch_dec_input = []
        batch_dec_out = []

        for j in range(start, stop):
            a, b, c = self.dataset[j]
            batch_enc.append(a[0])
            batch_dec_input.append(b[0])
            batch_dec_out.append(c[0])

        
        batch_enc = (np.array(batch_enc))
        batch_dec_input = np.array(batch_dec_input)
        batch_dec_out = np.array(batch_dec_out)

        return [batch_enc, batch_dec_input], batch_dec_out

    def __len__(self):
        return int(self.totl_points / self.batch_size)


train_dataset = Dataset(df_train, tk_inp, tk_out, 35)
train_dataloader = Dataloader(batch_size=512, dataset=train_dataset)


val_dataset = Dataset(df_val, tk_inp, tk_out, 35)
val_dataloader = Dataloader(batch_size=512, dataset=val_dataset)



class Encoder(tf.keras.layers.Layer):
   

    def __init__(self, vocab_size, emb_dims, enc_units, input_length, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units 
        self.embedding = layers.Embedding(vocab_size, emb_dims)
        self.lstm = layers.LSTM(self.enc_units, return_state=True, return_sequences=True)

    def call(self, enc_input, states):
        emb = self.embedding(enc_input)
        enc_output, state_h, state_c = self.lstm(emb, initial_state=states)
        return enc_output, state_h, state_c

    def initialize(self, batch_size):
        return tf.zeros(shape=(batch_size, self.enc_units)), tf.zeros(shape=(batch_size, self.enc_units))


# THIS IS ATTENTION LAYER FOR DOT MODEL
class Attention(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.units = units

    def call(self, enc_output, dec_state):
        dec_state = tf.expand_dims(dec_state, axis=-1)
        score = tf.matmul(enc_output, dec_state)
        att_weights = tf.nn.softmax(score, axis=1)
        context_vec = att_weights * enc_output
        context_vec = tf.reduce_sum(context_vec, axis=1)
        return context_vec, att_weights


class Onestepdecoder(tf.keras.Model):

    def __init__(self, vocab_size, emb_dims, dec_units, input_len, att_units, batch_size):
        super().__init__()
        self.emb = layers.Embedding(vocab_size, emb_dims, input_length=input_len)
        self.att = Attention(att_units)
        self.lstm = layers.LSTM(dec_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size, activation="softmax")

    def call(self, encoder_output, input, state_h, state_c):
        emb = self.emb(input)
        dec_output, dec_state_h, dec_state_c = self.lstm(emb, initial_state=[state_h, state_c])
        context_vec, alphas = self.att(encoder_output, dec_state_h)
        dense_input = tf.concat([tf.expand_dims(context_vec, 1), dec_output], axis=-1)
        fc = self.dense(dense_input)
        return fc, dec_state_h, dec_state_c, alphas


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, emb_dims, dec_units, input_len, att_units, batch_size):
        super().__init__()
        self.input_len = input_len
        self.onestepdecoder = Onestepdecoder(vocab_size, emb_dims, dec_units, input_len, att_units, batch_size)

    def call(self, dec_input, enc_output, state_h, state_c):
        current_state_h = state_h
        current_state_c = state_c
        pred = []
        alpha_values = []
        for i in range(self.input_len):
            current_vec = dec_input[:, i]
            current_vec = tf.expand_dims(current_vec, axis=-1)
            dec_output, dec_state_h, dec_state_c, alphas = self.onestepdecoder(enc_output, current_vec, current_state_h,
                                                                               current_state_c)
            current_state_h = dec_state_h
            current_state_c = dec_state_c

            pred.append(dec_output)
            alpha_values.append(alphas)
            
        output = tf.concat(pred, axis=1)
        alpha_values = tf.concat(alpha_values, axis=-1)
        return output, alpha_values


class encoder_decoder(tf.keras.Model):

    def __init__(self, enc_vocab_size, enc_emb_dim, enc_units, enc_input_length,
                 dec_vocab_size, dec_emb_dim, dec_units, dec_input_length, att_units, batch_size):

        super().__init__()
        self.batch_size = batch_size
        self.encoder = Encoder(enc_vocab_size, enc_emb_dim, enc_units, enc_input_length, batch_size)
        self.decoder = Decoder(dec_vocab_size, dec_emb_dim, dec_units, dec_input_length, att_units, batch_size)

    def call(self, data):
       
        inp1, inp2 = data
        enc_output, enc_state_h, enc_state_c = self.encoder(inp1, self.encoder.initialize(self.batch_size))
        dec_output, alphas = self.decoder(inp2, enc_output, enc_state_h, enc_state_c)
        return dec_output


model = encoder_decoder(enc_vocab_size=len(tk_inp.word_index) + 1,
                        enc_emb_dim=300,
                        enc_units=256, enc_input_length=35,
                        dec_vocab_size=len(tk_out.word_index) + 1,
                        dec_emb_dim=300,
                        dec_units=256,
                        dec_input_length=35,
                        att_units=256,
                        batch_size=512)

callback = [
    tf.keras.callbacks.ModelCheckpoint("C:/Users/Xavier/Desktop/Text to FSL/091423_tagalog.h5", save_best_only=True,
                                       mode="min",
                                       save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=0.0001),
    tf.keras.callbacks.TensorBoard("C:/Users/Xavier/Desktop/Text to FSL/logs/save", histogram_freq=1)
]

train_steps = train_dataloader.__len__()
val_steps = val_dataloader.__len__()
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy')

model.fit(train_dataloader, steps_per_epoch=train_steps, epochs=50, validation_data=val_dataloader,
          validation_steps=val_steps, callbacks=callback)
model.build([(512, 35), (512, 35)])
model.summary()


# LOADING THE WEIGHTS FOR BEST MODEL
model.load_weights("C:/Users/Xavier/Desktop/Text to FSL/091423_tagalog.h5")


def beam_search(input, model, k):
    seq = tk_inp.texts_to_sequences([input])
    seq = pad_sequences(seq, maxlen=35, padding="post")

    state = model.layers[0].initialize(1)
    enc_output, enc_state_h, enc_state_c = model.layers[0](seq, state)

    input_state_h = enc_state_h
    input_state_c = enc_state_c
    k_beams = [[tf.ones((1, 1), dtype=tf.int32), 0.0]]
    for i in range(35):
        candidates = []
        for sent_pred, prob in k_beams:
            if tk_out.word_index["<end>"] in sent_pred.numpy():
                candidates += [[sent_pred, prob]]
            else:

                dec_input = model.layers[1].layers[0].layers[0](sent_pred)
                dec_output, dec_state_h, dec_state_c = model.layers[1].layers[0].layers[2](dec_input,
                                                                                           initial_state=[input_state_h,
                                                                                                          input_state_c])

                context_vec, alphas = model.layers[1].layers[0].layers[1](enc_output, dec_state_h)

                dense_input = tf.concat([tf.expand_dims(context_vec, 1), tf.expand_dims(dec_state_h, 1)], axis=-1)
                dense = model.layers[1].layers[0].layers[3](dense_input)

                pred = tf.argsort(dense, direction='DESCENDING')[:, :, :k]
                for w in range(k):
                    candidates += [[tf.concat((sent_pred, pred[:, :, w]), axis=-1),
                                    (prob + tf.math.log(dense[:, :, pred[:, :, w][0][0]])[0][0])]]
        k_beams = sorted(candidates, key=lambda tup: tup[1], reverse=True)[:k]

    all_sent = []
    for i, score in k_beams:
        sent = ""
        for j in range(1, 35):
            sent += tk_out.index_word[i.numpy()[:, j][0]] + " "
            if tk_out.index_word[i.numpy()[:, j][0]] == "<end>":
                break
        all_sent.append((sent.strip(), score.numpy()))
    return all_sent


# BLEU SCORE
BLEU_beam = []
index = []
np.random.seed(1)
test_data = df_val.loc[np.random.choice(df_val.index, size=2000, replace=False)]
for ind, i in tqdm(test_data.iterrows(), position=0):
    try:
        pred = beam_search(str(i.enc_input), model, 3)[0][0].split()
        act = [str(i.dec_output).split()]
        b = bleu.sentence_bleu(act, pred)
        BLEU_beam.append(b)
    except:
        index.append(ind)

print("BLEU Score = ", np.mean(BLEU_beam))

print("INPUT SENTENCE ===> ",df_test.enc_input.values[19])
print("="*50)
print("ACTUAL OUTPUT ===> ",df_test.dec_output.values[19])
print("="*50)
print("BEAM SEARCH OUTPUT ,  SCORE")
# BEAM width = 3, change for a more accurate model but having longer time
bm = (beam_search(df_test.enc_input.values[19],model,3))
for i in bm:
    print(i)