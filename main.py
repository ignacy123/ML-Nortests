import tensorflow as tf
import numpy as np
import argparse
import sys
import pprint
from collections import Counter

from keras.layers import Dropout

LIMIT=10000000
EPOCHS=1000
BATCH_SIZE=128


parser = argparse.ArgumentParser()
parser.add_argument("--train", required = True)
parser.add_argument("--validation", required = True)
args = parser.parse_args()

with open(args.train) as f:
    train_count = int(f.readline())
    train_n = int(f.readline())
    train_distr = []
    train = np.zeros([train_count, train_n + 1])
    for i in range (train_count):
        line = f.readline()
        numbers, distr_name = line.rsplit(" ", 1)
        train_distr.append(distr_name[:-1])
        train[i, :] = np.fromstring(numbers, sep = " ", dtype=float)
    np.random.shuffle(train)
    train_res = train[:, -1]
    train = np.delete(train, np.s_[-1:], axis=1)

with open(args.validation) as f:
    val_count = int(f.readline())
    val_n = int(f.readline())
    val_distr = []
    val = np.zeros([val_count, val_n + 1])
    for i in range (val_count):
        line = f.readline()
        numbers, distr_name = line.rsplit(" ", 1)
        val_distr.append(distr_name[:-1])
        val[i, :] = np.fromstring(numbers, sep = " ", dtype=float)
    np.random.shuffle(val)
    val_res = val[:, -1]
    val = np.delete(val, np.s_[-1:], axis=1)


train = np.clip(train,-LIMIT,LIMIT)
val = np.clip(val,-LIMIT,LIMIT)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(128, input_dim=train_n))

model.add(tf.keras.layers.Dense(256, activation='leaky_relu', kernel_initializer='he_uniform'))

model.add(tf.keras.layers.Dense(128, activation='leaky_relu', kernel_initializer='he_uniform'))

model.add(tf.keras.layers.Dense(64, activation='leaky_relu', kernel_initializer='he_uniform'))

model.add(tf.keras.layers.Dense(16, activation='leaky_relu', kernel_initializer='he_uniform'))

model.add(tf.keras.layers.Dense(8, activation='leaky_relu', kernel_initializer='he_uniform'))

model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train, train_res, validation_data=(val, val_res), epochs=EPOCHS, batch_size=BATCH_SIZE)

correct = []
for res, expected, distr in zip(model.predict(val), val_res, val_distr):
    if (res >= 1/2 and expected) or  (res < 1/2 and not expected):
        correct.append(distr)

print("The model was trained on the following probes:")
pprint.pp(Counter(train_distr))
print("The model was validated on the following probes:")
pprint.pp(Counter(val_distr))
print("Validation probes prediction accuracy of the trained model:")
counter_distr = Counter(val_distr)
counter_correct = Counter(correct)
pprint.pp({key: str(int(counter_correct[key]/counter_distr[key] * 100)) + "%" for key in counter_distr})

print(f"Total Accuracy: { int(len(correct)/len(val) * 100) }%")

