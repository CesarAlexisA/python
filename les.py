import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report # Añadir classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE # Importar SMOTE
from collections import Counter # Para verificar la distribución

# ... (El resto de tus funciones load_data, build_model, plot_accuracy son las mismas) ...

def load_data(data_file):
    data = pd.read_csv(data_file)
    y = data['fast_charging_available'].values
    x = data.drop(['fast_charging_available', 'model'], axis=1)

    categorical_cols = x.select_dtypes(include=['object']).columns
    boolean_cols = x.select_dtypes(include=['bool']).columns
    numerical_cols = x.select_dtypes(include=np.number).columns

    for col in boolean_cols:
        x[col] = x[col].astype(int)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols.tolist() + boolean_cols.tolist()),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    x_processed = preprocessor.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_accuracy(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.close()


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data('Smartphones_cleaned_dataset.csv')

    print("Distribución original de la variable objetivo (y_train):")
    print(Counter(y_train))
    print("-" * 30)

    # --- Aplicar SMOTE para balancear el conjunto de entrenamiento ---
    print("Aplicando SMOTE para balancear el conjunto de entrenamiento...")
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    print("Distribución de la variable objetivo (y_train) después de SMOTE:")
    print(Counter(y_train_resampled))
    print("-" * 30)
    # --- Fin de la aplicación de SMOTE ---

    model = build_model(x_train_resampled.shape[1]) # Usa la forma de los datos remuestreados

    # Entrenar el modelo con los datos remuestreados.
    # Ya no se necesita 'class_weight' si los datos de entrenamiento ya están balanceados con SMOTE.
    history = model.fit(x_train_resampled, y_train_resampled, epochs=50, batch_size=32,
                        validation_split=0.2, verbose=1)

    plot_accuracy(history)

    y_pred = (model.predict(x_test) > 0.5).astype("int32")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0 (Sin Carga Rápida)', '1 (Con Carga Rápida)']) # Etiquetas para claridad
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Reporte de clasificación para ver Precision, Recall, F1-score por clase
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['0 (Sin Carga Rápida)', '1 (Con Carga Rápida)']))