
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def carregar_pasta(pasta):
    vetores = []

    arquivos = sorted([f for f in os.listdir(pasta) if f.endswith(".csv")])
    print(f"Arquivos encontrados em {pasta}: {len(arquivos)}")

    for arq in arquivos:
        df = pd.read_csv(os.path.join(pasta, arq))

        b0 = df["betti_0"].values
        b1 = df["betti_1"].values

        vetor = np.concatenate([b0, b1])
        vetores.append(vetor)

    return np.array(vetores)



pasta_1 = "/content/drive/MyDrive/Monografia /saídas/Curvas de Betti (B+)/csvs"
pasta_2 = "/content/drive/MyDrive/Monografia /saídas/Curvas de Betti (AB+)/csvs"



X_1 = carregar_pasta(pasta_1)
X_2 = carregar_pasta(pasta_2)

y_1 = np.zeros(X_1.shape[0])  
y_2 = np.ones(X_2.shape[0])   

X = np.vstack((X_1, X_2))
y = np.concatenate((y_1, y_2))



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


best_params = {
    'num_layers': 4,

    'units_0': 64,
    'activation_0': 'leaky_relu',
    'use_dropout_0': True,
    'rate_0': 0.1,

    'units_1': 96,
    'activation_1': 'tanh',
    'use_dropout_1': True,
    'rate_1': 0.1,

    'units_2': 192,
    'activation_2': 'linear',
    'use_dropout_2': True,
    'rate_2': 0.2,

    'units_3': 224,
    'activation_3': 'relu',
    'use_dropout_3': True,
    'rate_3': 0.3,

    'optimizer': 'SGD',
    'learning_rate': 0.01,
    'sgd_momentum': 0.0
}



def build_fixed_mlp(input_shape, params):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    for i in range(params['num_layers']):
        model.add(
            keras.layers.Dense(params[f'units_{i}'])
        )

        activation = params[f'activation_{i}']
        if activation == 'leaky_relu':
            model.add(keras.layers.LeakyReLU(alpha=0.1))
        else:
            model.add(keras.layers.Activation(activation))

        if params.get(f'use_dropout_{i}', False):
            model.add(
                keras.layers.Dropout(params[f'rate_{i}'])
            )

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    optimizer = keras.optimizers.SGD(
        learning_rate=params['learning_rate'],
        momentum=params['sgd_momentum']
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model



input_shape = (X_train.shape[1],)
model = build_fixed_mlp(input_shape, best_params)

model.summary()


history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=1
)


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nLoss final: {loss:.4f}")
print(f"Accuracy final: {accuracy:.4f}")


