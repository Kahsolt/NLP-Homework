#!/usr/bin/env python3
# Author: Armit
# Create Time: 2019/12/13 

import numpy as np
from random import random, randrange

from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split, cross_val_score
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam

if __name__ == "__main__":
  model = Sequential([
    Dense(32, activation='relu', input_dim=2),
    # Dropout(0.25),
    # Dense(8, activation='relu'),
    # Dropout(0.5),
    Dense(1),
  ])
  model.build()
  model.summary()
  adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
  model.compile(optimizer=adam, loss='mse', metrics=['accuracy', 'mse', 'mae'])

  # the f() to regress
  f = lambda x, y: 233 * x - 114.514 * y + randrange(10)
  
  X, y = [ ], [ ]
  for _ in range(10000000):
    a, b = randrange(100), randrange(100)
    X.append([a, b])
    y.append(f(a, b))
  X, y = np.array(X), np.array(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  model.fit(X_train, y_train, epochs=4, batch_size=64)

  score = model.evaluate(X_test, y_test)
  print("Scores: %r" % score)
