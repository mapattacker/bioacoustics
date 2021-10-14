"""refactored from https://www.kaggle.com/prajaktaparate14/audio-classification"""

from time import time

from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



class train:

    def __init__(self, input_shape, num_labels):
        """
        Args:
            input_shape (int): number of features
            num_labels (int): number of classes
        """

        self.input_shape = input_shape
        self.num_labels = num_labels


    def model_arch(self):
        """define NN architecture in keras"""

        model = Sequential()

        ### first Layer
        model.add(Dense(100, input_shape=(self.input_shape,)))
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
        model.add(Dense(self.num_labels))
        model.add(Activation('softmax'))
        model.compile(
                loss='categorical_crossentropy', 
                metrics=['accuracy'], 
                optimizer='adam')

        return model


    def start_train(X_train, X_test, y_train, y_test, epochs, batch_size):
        """start training"""

        model = self.model_arch()
        checkpointer = ModelCheckpoint(filepath='./', 
                            verbose=1, 
                            save_best_only=True)
        
        start = time()
        model.fit(X_train, y_train, 
                batch_size=batch_size, 
                epochs=epochs, 
                validation_data=(X_test, y_test), 
                callbacks=[checkpointer])

        print('Training completed in time: ', time()-start)
        test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(test_accuracy[1])



if __name__ == "__main__":
    epochs=20 
    batch_size=32
    input_shape=len(X_train[0])
    num_labels=len(np.unique(y_train))

    t = train(input_shape, num_labels)
    t.start_train(X_train, X_test, y_train, y_test, epochs, batch_size)
