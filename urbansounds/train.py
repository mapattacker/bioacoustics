"""refactored from https://www.kaggle.com/prajaktaparate14/audio-classification"""

from time import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam



class train:

    def __init__(self, input_shape, num_labels, 
                    model_path="./model", 
                    mapping_path="mapping.json"):
        """
        Args:
            input_shape (int): number of features
            num_labels (int): number of classes
        """

        self.input_shape = input_shape
        self.num_labels = num_labels
        self.model_path = model_path
        self.encoder_path = encoder_path


    def model_arch(self):
        """define NN architecture in keras"""

        model = Sequential()

        ### first Layer
        model.add(Dense(1024, input_shape=(self.input_shape,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        ### second Layer
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        ### third Layer
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        ### final Layer
        model.add(Dense(self.num_labels))
        model.add(Activation('softmax'))
        model.compile(
                loss='categorical_crossentropy', 
                metrics=['accuracy'], 
                optimizer='adam')

        return model


    def start_train(self, X_train, X_test, y_train, y_test, epochs, batch_size):
        """start training"""

        model = self.model_arch()
        checkpointer = ModelCheckpoint(filepath=self.model_path, 
                            verbose=1, 
                            save_best_only=True)
        
        start = time()
        model.fit(X_train, y_train, 
                batch_size=batch_size, 
                epochs=epochs, 
                validation_data=(X_test, y_test), 
                callbacks=[checkpointer])

        elapsed_time = round((time()-start)/60, 2)
        print(f'Training completed in time: {elapsed_time} min')
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f'Accuracy: {test_accuracy[1]}')


    def evaluate(self, X_test, cf_gradient=True):
        """get evaluation reports.
        
        Rets:
            (str): classification report. Use print().
            (df): confusion matrix. Do not use print().
        """

        model = load_model(self.model_path)
        with open(self.mapping_path) as f:
            mapping = json.load(f)
            classes = [mapping[i] for i in mapping]

        prediction = model.predict(X_test)
        y_pred = np.argmax(prediction, axis=1)
        y_test_orig = np.argmax(y_test, axis=1)

        # classification report
        class_report = classification_report(y_test_orig, y_pred, target_names=classes)
        
        # confusion matrix
        cfm = confusion_matrix(y_test_orig, y_pred)
        conf_matrix = pd.DataFrame(cfm, columns=classes, index=classes)
        if cf_gradient:
            conf_matrix = conf_matrix.style.background_gradient(cmap='coolwarm')
        
        return class_report, conf_matrix





if __name__ == "__main__":
    epochs=20 
    batch_size=32
    input_shape=len(X_train[0])
    num_labels=y_train.shape[1]

    t = train(input_shape, num_labels)
    t.start_train(X_train, X_test, y_train, y_test, epochs, batch_size)
    class_r, conf_m = t.evaluate(X_test, True)
    print(class_r)
    conf_m
