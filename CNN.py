import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



altura = 97
largura = 90



def carregar_imagens(pasta, label, limite=500):

    imagens = []
    labels = []

    arquivos = sorted(os.listdir(pasta))[:limite]

    for arq in arquivos:

        caminho = os.path.join(pasta, arq)

        img = keras.utils.load_img(
            caminho,
            color_mode='grayscale',
            target_size=(altura, largura)
        )

        img_array = keras.utils.img_to_array(img)

        imagens.append(img_array)
        labels.append(label)

    return np.array(imagens), np.array(labels)



pasta_1 = "/content/drive/MyDrive/Crislaine/digitais/A-"
pasta_2 = "/content/drive/MyDrive/Crislaine/digitais/O-"



X_1, y_1 = carregar_imagens(pasta_1, 0, limite=500)
X_2, y_2 = carregar_imagens(pasta_2, 1, limite=500)


X = np.concatenate([X_1, X_2], axis=0)
y = np.hstack([y_1, y_2])

print("Formato das imagens:", X.shape)
print("Formato dos labels:", y.shape)

print("\nClasses:")
print(np.unique(y, return_counts=True))



X = X / 255.0



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)



params = {

    'conv_layers': 2,

    'filters_0': 8,
    'kernel_0': 5,
    'activation_conv_0': 'sigmoid',
    'maxpooling_0': 4,

    'filters_1': 8,
    'kernel_1': 5,
    'activation_conv_1': 'sigmoid',
    'maxpooling_1': 2,

    'dense_layers': 1,

    'dense_units_0': 96,
    'activation_dense_0': 'leaky_relu',
    'dropout_0': False,
    'drop_rate_0': 0.4,

    'optimizer': 'Nadam',
    'learning_rate': 0.001,
    'sgd_momentum': 0.6
}



def build_model(params):

    model = keras.Sequential()

    model.add(
        keras.layers.Input(shape=(altura, largura, 1))
    )


    for i in range(params['conv_layers']):

        activation = params[f'activation_conv_{i}']


        model.add(
            keras.layers.Conv2D(
                filters=params[f'filters_{i}'],
                kernel_size=params[f'kernel_{i}'],
                padding='same'
            )
        )


        if activation == 'leaky_relu':

            model.add(
                keras.layers.LeakyReLU()
            )

        else:

            model.add(
                keras.layers.Activation(activation)
            )


        model.add(
            keras.layers.MaxPooling2D(
                pool_size=params[f'maxpooling_{i}']
            )
        )


    model.add(keras.layers.Flatten())


    model.add(
        keras.layers.Dense(
            params['dense_units_0']
        )
    )

    activation = params['activation_dense_0']

    if activation == 'leaky_relu':

        model.add(
            keras.layers.LeakyReLU()
        )

    else:

        model.add(
            keras.layers.Activation(activation)
        )


    if params['dropout_0']:

        model.add(
            keras.layers.Dropout(
                params['drop_rate_0']
            )
        )


    model.add(
        keras.layers.Dense(
            2,
            activation='softmax'
        )
    )


    if params['optimizer'] == 'Adam':

        optimizer = keras.optimizers.Adam(
            learning_rate=params['learning_rate']
        )

    elif params['optimizer'] == 'Nadam':

        optimizer = keras.optimizers.Nadam(
            learning_rate=params['learning_rate']
        )

    else:

        optimizer = keras.optimizers.SGD(
            learning_rate=params['learning_rate'],
            momentum=params['sgd_momentum']
        )


    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



model = build_model(params)

model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,

    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)



loss, acc = model.evaluate(X_test, y_test)

print(f"\nLoss: {loss:.4f}")
print(f"Accuracy: {acc:.4f}")



