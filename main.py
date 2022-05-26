import tensorflow as tf
import numpy as np
import argparse
import pprint
import random
from collections import Counter

LIMIT=10000000
EPOCHS=200
BATCH_SIZE=128


parser = argparse.ArgumentParser()
parser.add_argument("--train")
parser.add_argument("--validation", required = True)
parser.add_argument("--save-path")
parser.add_argument("--load-path")
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()

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
    val_res = val[:, -1]
    val = np.delete(val, np.s_[-1:], axis=1)
    val = np.clip(val,-LIMIT,LIMIT)

if args.train:
    with open(args.train) as f:
        train_count = int(f.readline())
        train_n = int(f.readline())
        train_distr = []
        train = np.zeros([train_count, train_n + 1])
        data = f.readlines()
        random.shuffle(data)
        for i, line in enumerate(data):
            numbers, distr_name = line.rsplit(" ", 1)
            train_distr.append(distr_name[:-1])
            train[i, :] = np.fromstring(numbers, sep = " ", dtype=float)
        train_res = train[:, -1]
        train = np.delete(train, np.s_[-1:], axis=1)
        train = np.clip(train,-LIMIT,LIMIT)

if args.load_path:
    model = tf.keras.models.load_model(args.load_path)
else:
    if not train_n:
        print("You need to specify a training file if you want to create a new model...")
        raise SystemExit(2)
    print("No model path specified, creating a new model...")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=train_n))
    model.add(tf.keras.layers.Dense(256, activation='leaky_relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(128, activation='leaky_relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(64, activation='leaky_relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(16, activation='leaky_relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(8, activation='leaky_relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

if train_n:
    print("The model will be trained on the following probes:")
    pprint.pp(Counter(train_distr))
    model.fit(train, train_res, validation_data=(val, val_res), epochs=EPOCHS, batch_size=BATCH_SIZE)

if args.debug:
    for layer in model.layers: 
        print(f"Layer: { layer.get_config() }")
        print(f"Min weight: { min([weights.min() for weights in layer.get_weights()]) }, max weight: { max([weights.max() for weights in layer.get_weights()]) }")

if args.save_path:
    model.save(args.save_path)

correct = []
for res, expected, distr in zip(model.predict(val), val_res, val_distr):
    if (res >= 1/2 and expected) or  (res < 1/2 and not expected):
        correct.append(distr)
        
counter_distr = Counter(val_distr)
counter_correct = Counter(correct)
print("The model was validated on the following probes:")
pprint.pp(counter_distr)
print("Validation probes prediction accuracy of the trained model:")
pprint.pp({key: str(int(counter_correct[key]/counter_distr[key] * 100)) + "%" for key in counter_distr})

print(f"Total Accuracy: { int(len(correct)/len(val) * 100) }%")

