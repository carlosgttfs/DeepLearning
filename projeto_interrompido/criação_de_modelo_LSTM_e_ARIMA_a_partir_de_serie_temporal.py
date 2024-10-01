import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# Carregar os dados do arquivo Excel
file_path = 'H:\Meu Drive\Pesquisa de Mestrado\Volve production data.xlsx'
data = pd.read_excel(file_path)

# Verificar os nomes das colunas
print(data.columns)

# Exibir as primeiras linhas do DataFrame
print(data.head())

# Exibir informações sobre o DataFrame
print(data.info())

# Verificar se há valores positivos na coluna 'BORE_GAS_VOL'
print(data['BORE_GAS_VOL'].describe())

# Contar valores nulos nas colunas de interesse
print(data[['BORE_GAS_VOL', 'BORE_OIL_VOL', 'AVG_WHP_P']].isnull().sum())

# Ajustar o código de acordo com os nomes das colunas
# Substitua 'BORE_GAS_VOL', 'BORE_OIL_VOL' e 'AVG_WHP_P' pelos nomes corretos das colunas
data = data[(data['BORE_GAS_VOL'] > 0) & (data[['BORE_GAS_VOL', 'BORE_OIL_VOL', 'AVG_WHP_P']].notnull().all(axis=1))]

# Verificar a quantidade de dados após o filtro
print(f"Número de linhas após o filtro: {len(data)}")

# Divisão dos dados em variáveis de entrada (X) e saída (y)
X = data[['BORE_GAS_VOL', 'BORE_OIL_VOL', 'AVG_WHP_P']]
y = data['BORE_GAS_VOL']

# Verificar se X e y não estão vazios
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Dividir os dados em conjuntos de treinamento e teste
if len(X) > 0 and len(y) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Treinar o modelo ARIMA
    arima_model = ARIMA(y_train, order=(5, 1, 2))
    arima_model_fit = arima_model.fit()

    # Fazer previsões com o modelo ARIMA
    arima_predictions = arima_model_fit.forecast(steps=len(y_test))

    # Preparar os dados para LSTM
    X_train_lstm = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Construir o modelo LSTM
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    # Compilar o modelo
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    # Treinar o modelo LSTM
    lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32)

    # Fazer previsões com o modelo LSTM
    lstm_predictions = lstm_model.predict(X_test_lstm)

    # Avaliação do modelo ARIMA
    arima_rmse = mean_squared_error(y_test, arima_predictions, squared=False)
    arima_mae = mean_absolute_error(y_test, arima_predictions)

    # Avaliação do modelo LSTM
    lstm_rmse = mean_squared_error(y_test, lstm_predictions, squared=False)
    lstm_mae = mean_absolute_error(y_test, lstm_predictions)

    # Exibir resultados
    print(f'ARIMA - RMSE: {arima_rmse}, MAE: {arima_mae}')
    print(f'LSTM - RMSE: {lstm_rmse}, MAE: {lstm_mae}')
else:
    print("O DataFrame X ou y está vazio após o pré-processamento.")
