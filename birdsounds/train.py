"""refactored from the kaggle notebook
https://www.kaggle.com/shtrausslearning/keras-bird-spectogram-multiclass-classification/notebook
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,AveragePooling2D
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications import EfficientNetB4, ResNet50,ResNet101, VGG16, MobileNet, InceptionV3
import seaborn as sns; sns.set(style='whitegrid')



def HistPlot(history):
    """plot accuracy & loss of train/val"""

    fig, ax = plt.subplots(1,2,figsize=(12,4))
    sns.despine(top=True,left=True,bottom=True)

    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].grid(True,linestyle='--',alpha=0.5)
    
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'test'], loc='upper left')
    ax[1].grid(True,linestyle='--',alpha=0.5)
    plt.show()



def pretrained_model(head_id, LABELS, coefs):
    """Define model with different applications
    
    Args:
        head_id (str): vgg, resnet, mobilenet, inception, efficientnet
    """

    model = Sequential()
    weights = "imagenet"

    if(head_id is 'vgg'):
        model.add(VGG16(input_shape=(shape[0],shape[1],3),
                            pooling='avg',
                            classes=1000,
                            include_top=False,
                            weights=weights))

    elif(head_id is 'resnet'):
        model.add(ResNet101(include_top=False,
                               input_tensor=None,
                               input_shape=(shape[0],shape[1],3),
                               pooling='avg',
                               classes=100,
                               weights=weights))

    elif(head_id is 'mobilenet'):
        model.add(MobileNet(alpha=1.0,
                               depth_multiplier=1,
                               dropout=0.001,
                               include_top=False,
                               weights=weights,
                               input_tensor=None,
                               input_shape = (shape[0],shape[1],3),
                               pooling=None,
                               classes=1000))

    elif(head_id is 'inception'):
        # 75x75
        model.add(InceptionV3(input_shape = (shape[0],shape[1],3), 
                                                    include_top = False, 
                                                    weights=weights))

    elif(head_id is 'efficientnet'):
        model.add(EfficientNetB4(input_shape = (shape[0],shape[1],3), 
                                    include_top = False, 
                                    weights=weights))

    ''' Tail Model Part '''
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(len(LABELS),activation='softmax'))

    # # freeze main model coefficients
    model.layers[0].trainable = False
    model.summary()
    return model


def model_custom(LABELS, shape):

    model = tf.keras.Sequential([
        
        # First conv block
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', 
                            input_shape=(shape[0], shape[1],3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second conv block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)), 
        
        # Third conv block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)), 
        
        # Fourth conv block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Global pooling instead of flatten()
        tf.keras.layers.GlobalAveragePooling2D(), 
        
        # Dense block
        tf.keras.layers.Dense(256, activation='relu'),   
        tf.keras.layers.Dropout(0.5),  
        tf.keras.layers.Dense(256, activation='relu'),   
        tf.keras.layers.Dropout(0.5),
        
        # Classification layer
        tf.keras.layers.Dense(len(LABELS), activation='softmax')
    ])

    return model
