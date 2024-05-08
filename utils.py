import pandas as pd
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.seasonal import STL

from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import statsmodels as st

import itertools

from sklearn.preprocessing import power_transform
from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import normaltest

import pmdarima as pm

import tensorflow as tf

import optuna

# Ignorar warnings
import warnings
warnings.filterwarnings("ignore")

class LinearForecast:
    """Flujo de datos para realizar un forecast

    Attributes:
        timeseries (pandas.DataFrame): [Serie de tiempo]
        x_label (str): [Etiqueta del eje x]
        y_label (str): [Etiqueta del eje y]
        sarimax_params (list): [Lista de combinación de parámetros del modelo SARIMAX]
        scaler (sklearn.Scaler): [Escalador de datos]
    """
    # lectura de datos
    def __init__(self, timeseries, x_label='Time', y_label='Value'):
        """Constructor que lee los datos directamente

        Lee los datos como un DataFrame de pandas

        Parameters
        ----------
        timeseries : pandas.DataFrame
            Datos en formato DataFrame.
        x_label : str [optional, default='Time']
            Nombre de la etiqueta x_label.
        y_label : str [optional, default='Value']
            Nombre de la etiqueta y_label."""

        self.x_label = x_label
        self.y_label = y_label
        self.timeseries = timeseries

    def set_x_label(self, x_label='Time'):
        """Actualiza el valor del atributo x_label

        Parameters
        ----------
        x_label : str [optional, default='Time']
            Nombre de la etiqueta x_label"""
        self.x_label = x_label

    def set_y_label(self, y_label='Value'):
        """Actualiza el valor del atributo y_label

        Parameters
        ----------
        y_label : str [optional, default='Value']
            Nombre de la etiqueta y_label"""
        self.y_label = y_label

    def split_dataset(self, timeseries=None, train_size=0.8):
        """Separa la serie de tiempo en train y test

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a separar.
        train_size : float [optional, default=0.8]
            Número de [0, 1] que representa el porcentaje de datos para el dataset de entrenamiento.

        Returns
        ----------
        tuple(pandas.DataFrame, pandas.DataFrame) con los datos en train y test respectivamente"""

        if timeseries is None:
            timeseries = self.timeseries

        i = int(timeseries.shape[0]*train_size)

        train = timeseries[:i]
        test = timeseries[i:]
        return train, test

    # Análisis
    def adf_test(self, timeseries=None) -> None:
        """Visualiza los resultados de la prueba de Dickey-Fuller

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a evaluar"""

        if timeseries is None:
            timeseries = self.timeseries

        print("Results of Dickey-Fuller Test:")
        dftest = adfuller(timeseries, autolag="AIC")
        dfoutput = pd.Series(
            dftest[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "#Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        print(dfoutput)

        if (dftest[1] <= 0.05) & (dftest[4]['5%'] > dftest[0]):
            print("\u001b[32mStationary\u001b[0m")
        else:
            print("\x1b[31mNon-stationary\x1b[0m")


    def plot_acf_pacf(self, timeseries=None, figsize=(8,5), **kwargs) -> None:
        """Visualiza autocorrelación y autocorrelación parcial

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a graficar
        **kwargs : dict [optional, default=dict]
            Argumentos extra para los métodos de plot_acf y plot_pacf."""

        if timeseries is None:
            timeseries = self.timeseries

        f = plt.figure(figsize=figsize)

        # Autocorrelación
        ax1 = f.add_subplot(121)
        plot_acf(timeseries, zero=False, ax=ax1, **kwargs)

        # Autocorrelación parcial
        ax2 = f.add_subplot(122)
        plot_pacf(timeseries, zero=False, ax=ax2, method='ols', **kwargs)

        plt.show()

    def plot_timeseries(self, timeseries=None, **kwargs) -> None:
        """Visualiza la serie de tiempo

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a gráficar
        **kwargs : dict [optional, default=dict]
            Argumentos extra para el método de plot."""

        if timeseries is None:
            timeseries = self.timeseries

        timeseries.plot(**kwargs) # pasamos los argumentos
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()


    def decompose(self, timeseries=None, periods_seasonality=(12,)) -> None:
        """Descompone la serie de tiempo

        Descompone la serie de tiempo con el método Multiple Seasonal-Trend decomposition

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a descomponer.
        periods_seasonality : tuple(int) [optional, default=(12,)]
            Los periodos estacionales en los cuales se descompondrá la serie"""

        if timeseries is None:
            timeseries = self.timeseries

        stl_kwargs = {"seasonal_deg": 0}
        model = MSTL(timeseries, periods=periods_seasonality, stl_kwargs=stl_kwargs)

        return model.fit()


    def plot_decomposition(self, timeseries=None, periods_seasonality=(12,)) -> None:
        """Visualiza la serie de tiempo y sus componentes

        Visualización de la serie de tiempo, su tendencia, estacionalidad y residuos.

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a gráficar.
        periods_seasonality : tuple(int) [optional, default=(12,)]
            Los periodos estacionales en los cuales se descompondrá la serie"""

        if timeseries is None:
            timeseries = self.timeseries

        data1 = timeseries.copy()

        res2 = self.decompose(data1, periods_seasonality)

        # Gráfica de descomposición
        fig, ax = plt.subplots(len(periods_seasonality) + 3, 1, sharex=True, figsize=(8, 8))

        plt.xlabel(self.x_label)

        res2.observed.plot(ax=ax[0])
        ax[0].set_ylabel('Observado')

        res2.trend.plot(ax=ax[1])
        ax[1].set_ylabel('Tendencia')

        if len(periods_seasonality) == 1:
            i = 0
            res2.seasonal.plot(ax=ax[2 + i])
            ax[2].set_ylabel('Estaciconal')
        else:
            for i in range(len(periods_seasonality)):
                res2.seasonal[f'seasonal_{periods_seasonality[i]}'].plot(ax=ax[2 + i])
                ax[2 + i].set_ylabel(f'Estaciconal_{periods_seasonality[i]}')

        res2.resid.plot(ax=ax[3 + i])
        ax[3 + i].set_ylabel('Residuos')

        fig.tight_layout()


    def transform(self, timeseries=None, method = 'log'):
        """Transforma la serie de tiempo

        Transforma la serie de tiempo con el método especificado.

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a transformar.
        method : str [optional, default='log']
            El método a utilizar para transformar:'log', 'sqrt', 'box-cox', 'yeo-johnson',
            'min_max_scaler' o 'standard_scaler'.

        Returns
        ----------
        pandas.DataFrame de los datos transformados con el método especificado
        """

        if timeseries is None:
            timeseries = self.timeseries

        timeseries = timeseries.copy()

        if method == 'log':
            return np.log(timeseries - timeseries.min() + 1)
        elif method == 'sqrt':
            return np.sqrt(timeseries)
        elif method == 'box-cox' or method == 'yeo-johnson':
            vals = power_transform(timeseries.iloc[:, 0].to_numpy().reshape(-1, 1), method=method)
            timeseries.iloc[:, 0] = vals.squeeze()
            return timeseries
        elif method == 'min_max_scaler':
            scaler = MinMaxScaler()
            vals = scaler.fit_transform(timeseries.iloc[:, 0].to_numpy().reshape(-1, 1))
            timeseries.iloc[:, 0] = vals
            self.scaler = scaler # Guarda el escalador para su posterior uso
            return timeseries
        elif method == 'standard_scaler':
            scaler = StandardScaler()
            vals = scaler.fit_transform(timeseries.iloc[:, 0].to_numpy().reshape(-1, 1))
            timeseries.iloc[:, 0] = vals
            self.scaler = scaler # Guarda el escalador para su posterior uso
            return timeseries
        else:
            raise Exception("Método no especificado")

    def test_normality(self, timeseries=None, threshold=0.05):
        """Evalua la normalidad de la distribución de los datos

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a evaluar.
        threshold : float [optional, default=0.05]
            Valor a partir del cual no la consideramos normal

        Returns
        ---------
        Valor booleano, False si no es normal, True si sí se considera normal"""
        if timeseries is None:
            timeseries = self.timeseries

        if(normaltest(timeseries)[1] < threshold):
            return False
        return True

    def plot_distribution(self, timeseries=None, **kwargs):
        """Visualiza la distribución de los datos de la serie de tiempo

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a gráficar
        **kwargs : dict [optional, default=dict]
            Argumentos extra para el método de plot."""

        if timeseries is None:
            timeseries = self.timeseries

        fig, ax = plt.subplots(1, 2, **kwargs)

        timeseries.hist(bins=50, ax=ax[0])

        timeseries.boxplot(ax=ax[1])

        plt.show()

    # Ajuste de parámetros
    def combine_parameters(self, p, d, q, P, D, Q, S, t):
        """Crea una combinación de posibles parámetros

        Genera una combinación con posibles valores de parámetros para un modelo SARIMAX

        Parameters
        ----------
        p : tuple(int)
            Parte autorregresiva de la parte no estacional de la serie de tiempo.
        d : tuple(int)
            Número de veces que se diferencia la parte no estacional de la serie de tiempo.
        q : tuple(int)
            Parte de medias móviles de la parte no estacional de la serie de tiempo.
        P : tuple(int)
            Parte autorregresiva de la parte estacional de la serie de tiempo.
        D : tuple(int)
            Número de veces que se diferencia la parte estacional de la serie de tiempo.
        Q : tuple(int)
            Parte de medias móviles de la parte estacional de la serie de tiempo.
        S : tuple(int)
            Estacionalidad de la parte estacional de la serie de tiempo.
        t : tuple(str)
            Tipo de tendencia: 'n', 'c', 't' y 'ct'.

        Returns
        ----------
        Lista de posibles combinaciones de los valores"""

        # Explorar función itertools.product
        no_estacional = list(itertools.product(p, d, q))
        estacional = list(itertools.product(P, D, Q, S))

        # Diferentes conmbinaciones
        self.sarimax_params = list(itertools.product(no_estacional, estacional, t))

        return self.sarimax_params

    def search_best_set_params(self, timeseries=None, sorting='AIC', ascending=True, sarimax_params=None):
        """Evaluación de modelo con las combinaciones del método sarimax_params

        ---Antes de usar se debe correr LinearForecast.sarimax_params---
        Con la combinación de posibles valores de parámetros para un modelo SARIMAX genera
        una tabla de resultados de las métricas AIC, BIC y LLF.

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo con la que se realiza el modelo.
        sorting : str [optional, default='AIC']
            Con que valor posible entre 'AIC', 'BIC' y 'LLF' desea que se ordene la tabla.
        ascending : bool  [optional, default=True]
            Orden en el que se regresa el DataFrame, True para ascendente y False para descendente.
        sarimax_params : list(tuple, tuple, str) [optional, default=LinearForecast.sarimax_params]
            Lista de parámetros a evaluar

        Returns
        ----------
        pandas.DataFrame de resultados con posibles combinaciones"""

        if timeseries is None:
            timeseries = self.timeseries

        if sarimax_params is None:
            sarimax_params = self.sarimax_params

        resultados = pd.DataFrame(columns=['params', 'AIC', 'BIC', 'LLF'])

        i=0
        for non_seasonal, seasonal, trend in sarimax_params:
            mod = SARIMAX(
                endog=timeseries,
                trend=trend,
                order=non_seasonal,
                seasonal_order=seasonal
            )
            try:
                results = mod.fit(disp=False)
    
                resultados.loc[i, 'params'] = str((non_seasonal, seasonal, trend))
                resultados.loc[i,'AIC'] = results.aic
                resultados.loc[i,'BIC'] = results.bic
                resultados.loc[i,'LLF'] = results.llf
                i += 1
            except:
                pass

        return resultados.sort_values(by=[sorting], ascending=ascending)


    def fit(self, timeseries=None, order=(0, 0, 0), seasonal_order=(0, 0, 0), trend='n'):
        """Ajuste de modelo SARIMAX

        Ajusta el modelo con los parámetros especificados y la serie de tiempo.

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo con la que se realiza el modelo.
        order : tuple(int, int, int) [optional, default=(0, 0, 0)]
            Valores p, d y f para la parte no estacional del modelo.
        seasonal_order : tuple(int, int, int) [optional, default=(0, 0, 0)]
            Valores P, D, F y S para la parte estacional del modelo.
        trend : tuple(str) [optional, default='n']
            Tipo de tendencia: 'n', 'c', 't' y 'ct'.

        Returns
        ----------
        Modelo statsmodels.SARIMAX ajustado"""
        if timeseries is None:
            timeseries = self.timeseries

        model = SARIMAX(
            endog = timeseries,
            trend = trend,
            order = order,
            seasonal_order = seasonal_order
        )

        results = model.fit()

        self.model = results

        return results

    # Predicción
    def predict(self, **kwargs):
        """Predicción de valores

        Parameters
        ----------
        **kwargs : dict [optional, default=dict]
            Argumentos para el método get_prediction del modelo

        Returns
        ----------
        statsmodels.PredictionResultsWrapper
        """

        return self.model.get_prediction(**kwargs)

    def predict_iter(self, timeseries=None, horizon=10, order=(0, 0, 0), seasonal_order=(0, 0, 0), trend='n'):
        """Predicción de modelo SARIMAX de forma iterativa

        Predice la serie de tiempo de forma iterativa a un horizonte al futuro

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo con la que se realiza el modelo y predicción.
        horizon : int [optional, default=10]
            Cantidad de valores que predecir al futuro.
        order : tuple(int, int, int) [optional, default=(0, 0, 0)]
            Valores p, d y f para la parte no estacional del modelo.
        seasonal_order : tuple(int, int, int) [optional, default=(0, 0, 0)]
            Valores P, D, F y S para la parte estacional del modelo.
        trend : tuple(str) [optional, default='n']
            Tipo de tendencia: 'n', 'c', 't' y 'ct'.

        Returns
        ----------
        Lista con valores predichos"""
        if timeseries is None:
            timeseries = self.timeseries

        timeseries = timeseries.copy()

        y_pred = []

        for i in range(horizon):
            mod = SARIMAX(
            endog=timeseries,
            trend=trend,
            order=order,
            seasonal_order=seasonal_order
            )

            results = mod.fit(disp=False)

            prediction = results.predict(timeseries.shape[0])
            timeseries = np.append(timeseries, prediction[0])
            y_pred.append(prediction[0])

        return y_pred


    # Métricas de errores
    def evaluate(self, y, predicted):
        """Evaluación del performance del modelo

        Calcula las métricas de error.

        Parameters
        ----------
        y : pandas.DataFrame
            La serie de tiempo con los valores reales.
        predicted : pandas.DataFrame
            La serie de tiempo con las predicciones.

        Returns
        ----------
        Diccionario con los resultados de las evaluaciones."""

        sub = (y - predicted).abs()

        mape = (sub / y.abs()).mean()
        mad = sub.mean()

        return {'MAPE': mape, 'MAD': mad}







# Código de solución estudiante 1
class NNForecast:
    """Flujo de datos para realizar un forecast

    Attributes:
        timeseries (pandas.DataFrame): [Serie de tiempo]
        x_label (str): [Etiqueta del eje x]
        y_label (str): [Etiqueta del eje y]
        scaler (sklearn.Scaler): [Escalador de datos]
    """
    # lectura de datos
    def __init__(self, timeseries, x_label='Time', y_label='Value'):
        """Constructor que lee los datos directamente

        Lee los datos como un DataFrame de pandas

        Parameters
        ----------
        timeseries : pandas.DataFrame
            Datos en formato DataFrame.
        x_label : str [optional, default='Time']
            Nombre de la etiqueta x_label.
        y_label : str [optional, default='Value']
            Nombre de la etiqueta y_label."""

        self.x_label = x_label
        self.y_label = y_label
        self.timeseries = timeseries

    def set_x_label(self, x_label='Time'):
        """Actualiza el valor del atributo x_label

        Parameters
        ----------
        x_label : str [optional, default='Time']
            Nombre de la etiqueta x_label"""
        self.x_label = x_label

    def set_y_label(self, y_label='Value'):
        """Actualiza el valor del atributo y_label

        Parameters
        ----------
        y_label : str [optional, default='Value']
            Nombre de la etiqueta y_label"""
        self.y_label = y_label

    def split_dataset(self, timeseries=None, train_size=0.8, validation_size=0.1):
        """Separa la serie de tiempo en train y test

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a separar.
        train_size : float [optional, default=0.8]
            Número de [0, 1] que representa el porcentaje de datos para el dataset de entrenamiento.

        Returns
        ----------
        tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame) con los datos en train, validation y test respectivamente"""

        if timeseries is None:
            timeseries = self.timeseries

        i_train = int(timeseries.shape[0]*train_size)
        i_val = int(timeseries.shape[0]*validation_size) + i_train

        train = timeseries[:i_train]
        validation = timeseries[i_train:i_val]
        test = timeseries[i_val:]
        return train, validation, test

    # Análisis
    def adf_test(self, timeseries=None) -> None:
        """Visualiza los resultados de la prueba de Dickey-Fuller

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a evaluar"""

        if timeseries is None:
            timeseries = self.timeseries

        print("Results of Dickey-Fuller Test:")
        dftest = adfuller(timeseries, autolag="AIC")
        dfoutput = pd.Series(
            dftest[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "#Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        print(dfoutput)

        if (dftest[1] <= 0.05) & (dftest[4]['5%'] > dftest[0]):
            print("\u001b[32mStationary\u001b[0m")
        else:
            print("\x1b[31mNon-stationary\x1b[0m")


    def plot_acf_pacf(self, timeseries=None, figsize=(8,5), **kwargs) -> None:
        """Visualiza autocorrelación y autocorrelación parcial

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a graficar
        **kwargs : dict [optional, default=dict]
            Argumentos extra para los métodos de plot_acf y plot_pacf."""

        if timeseries is None:
            timeseries = self.timeseries

        f = plt.figure(figsize=figsize)

        # Autocorrelación
        ax1 = f.add_subplot(121)
        plot_acf(timeseries, zero=False, ax=ax1, **kwargs)

        # Autocorrelación parcial
        ax2 = f.add_subplot(122)
        plot_pacf(timeseries, zero=False, ax=ax2, method='ols', **kwargs)

        plt.show()

    def plot_timeseries(self, timeseries=None, **kwargs) -> None:
        """Visualiza la serie de tiempo

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a gráficar
        **kwargs : dict [optional, default=dict]
            Argumentos extra para el método de plot."""

        if timeseries is None:
            timeseries = self.timeseries

        timeseries.plot(**kwargs) # pasamos los argumentos
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()


    def decompose(self, timeseries=None, periods_seasonality=(12,)) -> None:
        """Descompone la serie de tiempo

        Descompone la serie de tiempo con el método Multiple Seasonal-Trend decomposition

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a descomponer.
        periods_seasonality : tuple(int) [optional, default=(12,)]
            Los periodos estacionales en los cuales se descompondrá la serie"""

        if timeseries is None:
            timeseries = self.timeseries

        stl_kwargs = {"seasonal_deg": 0}
        model = MSTL(timeseries, periods=periods_seasonality, stl_kwargs=stl_kwargs)

        return model.fit()


    def plot_decomposition(self, timeseries=None, periods_seasonality=(12,)) -> None:
        """Visualiza la serie de tiempo y sus componentes

        Visualización de la serie de tiempo, su tendencia, estacionalidad y residuos.

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a gráficar.
        periods_seasonality : tuple(int) [optional, default=(12,)]
            Los periodos estacionales en los cuales se descompondrá la serie"""

        if timeseries is None:
            timeseries = self.timeseries

        data1 = timeseries.copy()

        res2 = self.decompose(data1, periods_seasonality)

        # Gráfica de descomposición
        fig, ax = plt.subplots(len(periods_seasonality) + 3, 1, sharex=True, figsize=(8, 8))

        plt.xlabel(self.x_label)

        res2.observed.plot(ax=ax[0])
        ax[0].set_ylabel('Observado')

        res2.trend.plot(ax=ax[1])
        ax[1].set_ylabel('Tendencia')

        if len(periods_seasonality) == 1:
            i = 0
            res2.seasonal.plot(ax=ax[2 + i])
            ax[2].set_ylabel('Estaciconal')
        else:
            for i in range(len(periods_seasonality)):
                res2.seasonal[f'seasonal_{periods_seasonality[i]}'].plot(ax=ax[2 + i])
                ax[2 + i].set_ylabel(f'Estaciconal_{periods_seasonality[i]}')

        res2.resid.plot(ax=ax[3 + i])
        ax[3 + i].set_ylabel('Residuos')

        fig.tight_layout()


    def transform(self, timeseries=None, method = 'log'):
        """Transforma la serie de tiempo

        Transforma la serie de tiempo con el método especificado.

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a transformar.
        method : str [optional, default='log']
            El método a utilizar para transformar:'log', 'sqrt', 'box-cox', 'yeo-johnson',
            'min_max_scaler' o 'standard_scaler'.

        Returns
        ----------
        pandas.DataFrame de los datos transformados con el método especificado
        """

        if timeseries is None:
            timeseries = self.timeseries

        timeseries = timeseries.copy()

        if method == 'log':
            self.transf_name = method
            return np.log(timeseries - timeseries.min() + 1)
        elif method == 'sqrt':
            self.transf_name = method
            return np.sqrt(timeseries)
        elif method == 'box-cox' or method == 'yeo-johnson':
            self.transf_name = method
            vals = power_transform(timeseries.iloc[:, 0].to_numpy().reshape(-1, 1), method=method)
            timeseries.iloc[:, 0] = vals.squeeze()
            return timeseries
        elif method == 'min_max_scaler':
            scaler = MinMaxScaler()
            vals = scaler.fit_transform(timeseries.iloc[:, 0].to_numpy().reshape(-1, 1))
            timeseries.iloc[:, 0] = vals
            self.scaler = scaler # Guarda el escalador para su posterior uso
            return timeseries
        elif method == 'standard_scaler':
            scaler = StandardScaler()
            vals = scaler.fit_transform(timeseries.iloc[:, 0].to_numpy().reshape(-1, 1))
            timeseries.iloc[:, 0] = vals
            self.scaler = scaler # Guarda el escalador para su posterior uso
            return timeseries
        else:
            raise Exception("Método no especificado")

    def test_normality(self, timeseries=None, threshold=0.05):
        """Evalua la normalidad de la distribución de los datos

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a evaluar.
        threshold : float [optional, default=0.05]
            Valor a partir del cual no la consideramos normal

        Returns
        ---------
        Valor booleano, False si no es normal, True si sí se considera normal"""
        if timeseries is None:
            timeseries = self.timeseries

        if(normaltest(timeseries)[1] < threshold):
            return False
        return True

    def plot_distribution(self, timeseries=None, **kwargs):
        """Visualiza la distribución de los datos de la serie de tiempo

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a gráficar
        **kwargs : dict [optional, default=dict]
            Argumentos extra para el método de plot."""

        if timeseries is None:
            timeseries = self.timeseries

        fig, ax = plt.subplots(1, 2, **kwargs)

        timeseries.hist(bins=50, ax=ax[0])

        timeseries.boxplot(ax=ax[1])

        plt.show()



    # Métricas de errores
    def evaluate(self, y, predicted):
        """Evaluación del performance del modelo

        Calcula las métricas de error.

        Parameters
        ----------
        y : pandas.DataFrame
            La serie de tiempo con los valores reales.
        predicted : pandas.DataFrame
            La serie de tiempo con las predicciones.

        Returns
        ----------
        Diccionario con los resultados de las evaluaciones."""

        sub = (y - predicted).abs()

        mape = (sub / y.abs()).mean()
        mad = sub.mean()

        return {'MAPE': mape, 'MAD': mad}

    # dividir una secuencia univariada en muestras
    def split_sequence_m_step(self, timeseries=None, n_steps_in=3, n_steps_out=1, multivariate=False):
        """Separación de los datos por pasos

        Separa las series de tiempo por pasos temporales de entrada y de salida
        para poder hacer modelos supervisados con las series de tiempo.

        Parameters
        ----------
        timeseries : pandas.DataFrame [optional, default=LinearForecast.timeseries]
            La serie de tiempo a dividir
        n_steps_in : int [optional, 3]
            Pasos de entrada
        n_steps_out : int [optional, default=1]
            Pasos de salida
        multivariate : Bool [optional, default=False]
            Pasos de entrada

        Returns
        ----------
        tuple(pandas.DataFrame, pandas.DataFrame) con los datos en input y target respectivamente"""
        if timeseries is None:
            timeseries = self.timeseries

        X, y = list(), list()
        for i in range(len(timeseries)):
            # encontrar el final de este patrón
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out - 1

            # comprobar si estamos más allá de la secuencia
            if out_end_ix > len(timeseries):
                break
            # reunir partes de entrada y salida del patrón
            seq_x, seq_y = (timeseries[i:end_ix, :-1], timeseries[end_ix-1:out_end_ix, -1]) if multivariate else \
                            (timeseries[i:end_ix], timeseries[end_ix-1:out_end_ix])
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    # Modelos redes neuronales secuenciales multicapa, 3 en total,
    # donde cada uno tiene una capa densa más que el modelo anterior.
    def MLP_models(self, n_steps_in=3, n_steps_out=1, n_features=1, **kwargs):
        self.models_MLP = []
        for i in range(3):
            self.models_MLP.append(
                tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(n_steps_in * n_features, )),
                    *[tf.keras.layers.Dense(100, activation='relu') for j in range(i + 1)],
                    tf.keras.layers.Dense(100, activation='relu'),
                    tf.keras.layers.Dense(n_steps_out)
                ])
            )

            # Compilar el modelo
            self.models_MLP[i].compile(**kwargs)

    # Modelos redes neuronales convolucionales, 3 en total,
    # donde cada uno tiene una capa convolucional más que el modelo anterior.
    def CNN_models(self, n_steps_in=3, n_steps_out=1, n_features=1, padding='valid', **kwargs):
        self.models_CNN = []
        for i in range(3):
            self.models_CNN.append(
                tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(n_steps_in, n_features, )),
                    *[tf.keras.layers.Conv1D(64, 2, activation='relu', padding=padding) for j in range(i + 1)],
                    tf.keras.layers.MaxPooling1D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(100, activation='relu'),
                    tf.keras.layers.Dense(n_steps_out)
                ])
            )
            # Compilar el modelo
            self.models_CNN[i].compile(**kwargs)

    # Modelos redes neuronales recurrentes, 3 en total,
    # donde cada uno tiene una capa LSTM más que el modelo anterior, que se apilan.
    def RNN_models(self, n_steps_in=3, n_steps_out=1, n_features=1, **kwargs):
        self.models_RNN = []
        for i in range(3):
            self.models_RNN.append(
                tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(n_steps_in, n_features, )),
                    *[tf.keras.layers.LSTM(100, activation='relu', return_sequences=True,) for j in range(i + 1)],
                    tf.keras.layers.LSTM(100, activation='relu'),
                    tf.keras.layers.Dense(n_steps_out)
                ])
            )
            # Compilar el modelo
            self.models_RNN[i].compile(**kwargs)

    # Modelos redes neuronales CNN-LSTM, 3 en total,
    # donde cada uno tiene una capa densa más en cada subsecuencia
    # de entrada que el modelo anterior.
    def CNN_LSTM_models(self, n_steps_in=3, n_steps_out=1, n_features=1, n_seq=2, padding='valid' , **kwargs):
        self.models_CNN_LSTM = []
        for i in range(3):
            self.models_CNN_LSTM.append(
                tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(None, int(n_steps_in/n_seq), n_features, )),
                    *[ tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, 1, activation='relu', padding=padding)) for j in range(i + 1)],
                    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=n_seq, padding=padding)),
                    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
                    tf.keras.layers.LSTM(100, activation='relu'),
                    tf.keras.layers.Dense(n_steps_out)
                ])
            )
            # Compilar el modelo
            self.models_CNN_LSTM[i].compile(**kwargs)

    # Entrenamiento de las 3 arquitecturas de cada tipo de modelo
    def fit_models(self, tr_X, tr_y, val_X, val_y, n_seq=2, n_steps_in=3, n_features=1, **kwargs):
        self.history = {}

        self.history['MLP'] = [m.fit(tr_X.reshape(tr_X.shape[0], -1), tr_y, validation_data=(val_X.reshape(val_X.shape[0], -1), val_y), **kwargs).history for m in self.models_MLP]
        self.history['CNN'] = [m.fit(tr_X, tr_y, validation_data=(val_X, val_y), **kwargs).history for m in self.models_CNN]
        self.history['RNN'] = [m.fit(tr_X, tr_y, validation_data=(val_X, val_y), **kwargs).history for m in self.models_RNN]
        self.history['CNN_LSTM'] = [m.fit(tr_X.reshape((tr_X.shape[0], n_seq, int(n_steps_in / n_seq), n_features)),
                                                    tr_y, validation_data=(
                                                        val_X.reshape((val_X.shape[0], n_seq, int(n_steps_in / n_seq), n_features)),
                                                        val_y),
                                                    **kwargs) for m in self.models_CNN_LSTM]

        return self.history
