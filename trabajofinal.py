import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
data = pd.read_csv('pokemon_data.csv')

# Calcular la mediana de 'base_experience' para crear una etiqueta binaria
median_experience = data['base_experience'].median()
data['label'] = (data['base_experience'] > median_experience).astype(int)

# Función para analizar la columna 'stats' y extraer los valores numéricos
def parse_stats(stats_str):
    stats_dict = {}
    if pd.isna(stats_str):
        return {}
    pairs = stats_str.split(', ')
    for pair in pairs:
        key, value = pair.split('=')
        stats_dict[key] = int(value)
    return stats_dict

# Aplicar la función para extraer las estadísticas en nuevas columnas
stats_df = data['stats'].apply(parse_stats).apply(pd.Series)
data = pd.concat([data, stats_df], axis=1)

# Seleccionar las columnas numéricas relevantes como características (X)
# Se incluyen 'height', 'weight' y las estadísticas extraídas.
feature_columns = ['height', 'weight', 'hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']
X = data[feature_columns].values
y = data['label'].values # La columna 'label' recién creada es la variable objetivo (y)

# Definir la función para cargar y preprocesar los datos
def load_data(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

# Definir la función para construir el modelo de red neuronal
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'), # Capa de salida para clasificación binaria
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', # Función de pérdida para clasificación binaria
                  metrics=['accuracy'])
    return model

# Definir la función para graficar la precisión de entrenamiento y validación
def plot_accuracy(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png') # Guarda el gráfico como imagen
    plt.close()

# Bloque principal de ejecución
if __name__ == '__main__':
    # Cargar y preprocesar los datos
    x_train, x_test, y_train, y_test = load_data(X, y)

    # Construir el modelo
    model = build_model(x_train.shape[1])

    # Entrenar el modelo
    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Generar y guardar el gráfico de precisión
    plot_accuracy(history)

    # Realizar predicciones y calcular la matriz de confusión
    y_pred = (model.predict(x_test, verbose=0) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Graficar y guardar la matriz de confusión
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png') # Guarda el gráfico como imagen
    plt.close()