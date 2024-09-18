
import numpy as np
from cmb_eval import cmb_eval

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
from tensorflow.keras.metrics import F1Score
import tensorflow as tf
import random
import os

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def dnn_classifer(moda, fold_name, train_data, val_data, test_data, num_classes, train = False, verbose = 1):

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    tf.config.experimental.enable_op_determinism()

    X_train, y_train = train_data 
    X_val, y_val = val_data
    X_test, y_test = test_data

    window_size = np.shape(X_train)[1]
    n_features = np.shape(X_train)[2]

    batch_size = 16
    epochs = 40

    input_shape = (window_size, n_features)

    model = Sequential()

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(300, activation='relu'))
    model.add(BatchNormalization()) # Necessary
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #             optimizer=keras.optimizers.legacy.Adam(),
    #             metrics=['accuracy'])

    # Compile the model with F1 score as a metric
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.legacy.Adam(),
                metrics=['accuracy', F1Score()])
    

    current_dir = os.path.join(os.path.dirname(__file__))  # Folder
    weight_dir = os.path.join(current_dir, 'weights', moda, fold_name + '.h5')
    
    if train == True:

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        print(np.shape(X_train))

        checkpoint = ModelCheckpoint(filepath=weight_dir, 
                             monitor='val_loss', 
                             save_best_only=True, 
                             mode='min')

        history = model.fit(X_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=(X_val, y_val), 
                            # class_weight=class_weight_dict,
                            callbacks=[early_stop, checkpoint],
                            verbose = verbose)
        # model.save(weight_dir)
    else:
        model.load_weights(weight_dir)

    
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    y_pred += 1
    y_test += 1

    acc_dict, prec_dict, recal_dict, f1_dict = cmb_eval(y_pred, y_test)

    return acc_dict, prec_dict, recal_dict, f1_dict