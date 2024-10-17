import pandas as pd

# Carregar os dados do arquivo Excel
file_path = 'I:\Meu Drive\Pesquisa de Mestrado\Volve production data.xlsx'
data = pd.read_excel(file_path)

# Visualizar as primeiras linhas do dataset
#print(data.head())

# Filtrar dados válidos
data = data[data['BORE_GAS_VOL'] > 0]
data = data.dropna()

# Divisão dos dados em variáveis de entrada (X) e saída (y)
X = data[['BORE_GAS_VOL', 'BORE_OIL_VOL', 'AVG_DOWNHOLE_PRESSURE']]
y = data['BORE_GAS_VOL']

# Dividir os dados em conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from statsmodels.tsa.arima.model import ARIMA

# Treinar o modelo ARIMA
arima_model = ARIMA(y_train, order=(5, 1, 2))
arima_model_fit = arima_model.fit()

# Fazer previsões com o modelo ARIMA
arima_predictions = arima_model_fit.forecast(steps=len(y_test))


#comentario
