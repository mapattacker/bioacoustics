import matplotlib.pyplot as plt


def HistPlot():
    """plot training & validation accuracy & loss"""
    
    fig,ax = plt.subplots(1,2,figsize=(12,4))
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
