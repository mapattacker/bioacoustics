"""refactored from https://www.kaggle.com/prajaktaparate14/audio-classification"""

from time import time

from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



class train:

    def model_arch(self, input_shape, num_labels):
        """define NN architecture in keras"""

        model = Sequential()

        ### first Layer
        model.add(Dense(100, input_shape=(input_shape,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        ### second Layer
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        ### third Layer
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        ### final Layer
        model.add(Dense(num_labels))
        model.add(Activation('softmax'))
        model.compile(
                loss='categorical_crossentropy', 
                metrics=['accuracy'], 
                optimizer='adam')

        return model


    def train(X_train, X_test, y_train, y_test, epochs, batch_size):
        """start training"""

        start = time()
        checkpointer = ModelCheckpoint(filepath='model/', 
                            verbose=1, 
                            save_best_only=True)

        model.fit(X_train, y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_data=(X_test, y_test), 
            callbacks=[checkpointer])

        print('Training completed in time: ', time()-start)
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(test_accuracy[1])



if __name__ == "__main__":
    t = train()
    model = t.model_arch(40, 10)
    print(model.summary())
