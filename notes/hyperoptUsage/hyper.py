import os
import keras
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from hyperopt import hp
from hyperopt.pyll import scope
from keras import Input, layers
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import Model
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import datetime
from collections import Counter

from cv import parse_args
from cv.dnn import ObjectiveCV, CmdFactory
np.random.seed(777)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


class _CV(ObjectiveCV):
    def fit(self, sp, X_tr, X_tr_aux, y_tr, X_val, X_val_aux, y_val):
        keras.backend.clear_session()
        print(sp)

        ## Init model
        input_bands = Input(shape=X_tr.shape[1:])
        base_model = ResNet50(input_shape=input_bands.shape[1:],
                          input_tensor=input_bands,
                          include_top=False,
                          classes=1,
                          pooling='avg'
                          )
        x = base_model.output
        x = Dropout(sp['dropout1'])(x)
        if sp['batchnorm'] == 1:
            x = BatchNormalization()(x)

        x = Dense(sp['n_filter'], activation='relu')(x)
        x = LeakyReLU()(x)
        x = Dropout(sp['dropout2'])(x)
        x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='pred')(x)

        model = Model(input_bands, x)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=sp['lr']),
                      metrics=['accuracy'])
        
        batch_size = sp['batch_size']
                
        now = datetime.datetime.now()

        ## Train
        ### Warm up (train only classify layer )
        for layer in base_model.layers: 
            layer.trainable = False
            
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=sp['lr']),
                      metrics=['accuracy'])
            
        generator = ImageDataGenerator(
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   rotation_range=25,
                                   fill_mode='wrap'
                                   )
        iter_ = generator.flow(X_tr, range(len(X_tr)), batch_size=batch_size, seed=123)

        def datagen():
            while True:
                X_batch, idx = iter_.next()
                yield X_batch, y_tr[idx]

        best_weights = 'weights/weights.hdf5'
        model.fit_generator(
            generator=datagen(),
            steps_per_epoch=int(len(X_tr) / batch_size),
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[
                EarlyStopping(monitor='val_loss',
                              patience=25,
                              verbose=1),
                ModelCheckpoint(best_weights,
                                save_best_only=True,
                                monitor='val_loss'),
                keras.callbacks.TensorBoard(log_dir="logs/chipping_{:%Y%m%dT%H%M}".format(now),
                                            histogram_freq=0, write_graph=True, write_images=False),
            ])

        ### Train all model
        for layer in base_model.layers:
            layer.trainable = True

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=sp['lr']),
                      metrics=['accuracy'])
        
        model.fit_generator(
            generator=datagen(),
            steps_per_epoch=int(len(X_tr) / batch_size),
            epochs=150,
            validation_data=(X_val, y_val),
            callbacks=[
                EarlyStopping(monitor='val_loss',
                              patience=25,
                              verbose=1),
                ModelCheckpoint(best_weights,
                                save_best_only=True,
                                monitor='val_loss'),
            ])
        
        model.load_weights(filepath=best_weights)

        return model

def main():
    base_name = 'stg0/chipping_resnet50_slide'
    seed = 101_000

    args = parse_args()
    cmd = CmdFactory(_CV)
    
    x_ng_tr = np.load('../tmp/NG_tr.npy').astype(float)
    x_ng_val = np.load('../tmp/NG_val.npy').astype(float)

    x_ok_tr = np.load('../tmp/OK_tr.npy').astype(float)
    x_ok_val = np.load('../tmp/OK_val.npy').astype(float)

    y_ng_tr = np.repeat([1.], x_ng_tr.shape[0])
    y_ng_val = np.repeat([1.], x_ng_val.shape[0])
    y_ok_tr = np.repeat([0.], x_ok_tr.shape[0])
    y_ok_val = np.repeat([0.], x_ok_val.shape[0])

    X_tr = np.concatenate((x_ng_tr, x_ok_tr))
    X_val = np.concatenate((x_ng_val, x_ok_val))    
    y_tr = np.concatenate((y_ng_tr, y_ok_tr))
    y_val = np.concatenate((y_ng_val, y_ok_val))

    X_tr, y_tr = shuffle(X_tr, y_tr, random_state=0)
    X_val, y_val = shuffle(X_val, y_val, random_state=0)

    if not os.path.exists('artifacts/stg0'):
        os.makedirs('artifacts/stg0')

    if args.type == 'hpt':
        space = {
            # 'lr': 1e-3,
            # 'batch_size': 16,
            # 'dropout1': 0.1,
            # 'dropout2': 0.1,
            # 'batchnorm': 0,
            # 'n_filter': 512,
            'lr': hp.loguniform('lr', np.log(1e-8), np.log(1e-4)),
            'batch_size': scope.int(hp.quniform('batch_size', 2, 8, 1)),
            'dropout1': hp.uniform('dropout1', 0, 0.5),
            'dropout2': hp.uniform('dropout2', 0, 0.5),
            'batchnorm': hp.choice('batchnorm', [0, 1]),
            'n_filter': scope.int(hp.quniform('n_filter', 64*2, 1024, 1)),
        }
        cmd.hpt(
            train_data=(X_tr, np.zeros((X_tr.shape[0], 0)), y_tr),
            val_data=(X_val, y_val),
            space=space,
            use_aux=False,
            seed=seed,
            trials_file='artifacts/{}_trials.pickle'.format(base_name),
            max_iter=args.max_iter)
    elif args.type == 'pred':
        params = {
            'lr': 4.2224658237043676e-05,
            'batch_size': 10,
            'dropout1': 0.2691817524765299,
            'dropout2': 0.4109641863922158,
            'batchnorm': 0,
            'n_filter': 223,
        }
        cmd.pred(
            train_data=(X, np.zeros((X.shape[0], 0)), y),
            val_data=(X_val, y_val),
            test_data=(X_test, np.zeros((X_test.shape[0], 0))),
            params=params,
            seed=seed,
            out_tr='artifacts/{}_train.npy'.format(base_name),
            out_test='artifacts/{}_test.npy'.format(base_name),
            n_bags=args.n_bags)
    else:
        raise ValueError('type must be hpt or pred.')


if __name__ == '__main__':
    main()
