
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MLP_Binary_HyperModel_Expanded(kt.HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=self.input_shape))

        num_layers = hp.Int('num_layers', min_value=2, max_value=10, step=1)

        for i in range(num_layers):
            units = hp.Int(
                f'units_{i}',
                min_value=32,
                max_value=256,
                step=32
            )

            model.add(keras.layers.Dense(units))

            activation = hp.Choice(
                f'activation_{i}',
                ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'linear']
            )

            if activation == 'leaky_relu':
                model.add(keras.layers.LeakyReLU(alpha=0.1))
            else:
                model.add(keras.layers.Activation(activation))

            if hp.Boolean(f'use_dropout_{i}'):
                model.add(
                    keras.layers.Dropout(
                        rate=hp.Float(
                            f'rate_{i}',
                            min_value=0.1,
                            max_value=0.5,
                            step=0.1
                        )
                    )
                )

        model.add(keras.layers.Dense(1, activation='sigmoid'))

        optimizer_choice = hp.Choice('optimizer', ['Adam', 'Nadam', 'SGD'])
        lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

        if optimizer_choice == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_choice == 'Nadam':
            optimizer = keras.optimizers.Nadam(learning_rate=lr)
        else:
            optimizer = keras.optimizers.SGD(
                learning_rate=lr,
                momentum=hp.Float(
                    'sgd_momentum',
                    min_value=0.0,
                    max_value=0.9,
                    step=0.1
                )
            )

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model


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


pasta_a = "/content/drive/MyDrive/Monografia /saídas/Curvas de Betti (B+)/csvs"
pasta_o = "/content/drive/MyDrive/Monografia /saídas/Curvas de Betti (AB+)/csvs"

X_a = carregar_pasta(pasta_a)
X_o = carregar_pasta(pasta_o)

y_a = np.zeros(X_a.shape[0])
y_o = np.ones(X_o.shape[0])

X = np.vstack((X_a, X_o))
y = np.concatenate((y_a, y_o))



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



input_shape = (X_train.shape[1],)
hypermodel = MLP_Binary_HyperModel_Expanded(input_shape)

tuner = kt.BayesianOptimization(
    hypermodel,
    objective='val_accuracy',
    max_trials=150,
    executions_per_trial=1,
    overwrite=False,
    directory="/content/drive/MyDrive/Monografia /saídas/otimizacao B+ e AB+",
    project_name="hiperparametros"
)

tuner.search_space_summary()



tuner.search(
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
    ]
)

tuner.results_summary()

best_hps = tuner.get_best_hyperparameters(1)[0]
print("Melhores hiperparâmetros:")
print(best_hps.values)

best_model = tuner.get_best_models(1)[0]

best_model.fit(
    X_train,
    y_train,
    epochs=100,
    verbose=1
)

loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
