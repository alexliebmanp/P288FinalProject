import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras import optimizers
from keras.layers import LSTM
from keras.layers import Dense, Reshape
from keras.models import Sequential
import math
from math import sqrt

class AVERT_LSTM():
    """
    LSTM class for many-to-many forecasting of volcanic activity.

    Input data: Time-serires data in a Pandas DataFrame with Datetime (if not Datetime indexed, tries to make it so) index and n_features.

    Forecasting task: Predict all n_features for n_future steps given n_past steps.

    """

    def __init__(self, df, input_vars=['Eruption_Activity', 'VPCC_RSAM', 'VPPC_RSAM', 'VPNC_RSAM', 'VPRS_RSAM', 'VPNC_Intensity', 'CO2_ppm', 'lake_size'], predict_vars=['Eruption_Activity', 'VPCC_RSAM', 'VPPC_RSAM', 'VPNC_RSAM', 'VPRS_RSAM', 'VPNC_Intensity', 'CO2_ppm', 'lake_size']):
        """

        Load data and preprocess by normalizing and scaling.

        args:
            - df:               (DataFrame) contains time-series data with Datetime index
            - input_vars:       (list) names of columns in df to use as input to LSTM
            - predict_vars:     (list) names of columns in df to predict using LSTM.
        """

        # filter data to columns we want in LSTM and handle any nans
        df.index = pd.to_datetime(df.index) # make sure index is datatime
        df.fillna(0, inplace=True)
        df_x = df[input_vars]
        df_y = df[predict_vars]
        self.df_x = df_x # save input DataFrame
        self.df_y = df_y # save predict DataFrame
        self.df = df[list(set(input_vars+predict_vars))] # combined DataFrame
        
        # preprocess data into both dataframes and numpy arrays
        self.df_x_scaled, self.x_scaler = self.NormAndScale(df_x)
        self.df_y_scaled, self.y_scaler = self.NormAndScale(df_y)
        self.x_data = self.df_x_scaled.to_numpy()
        self.y_data = self.df_y_scaled.to_numpy()
        self.index = df_x.index

    def CreateModel(self, n_past=1, n_future=1, n_divide=0.8, n_neurons=400, n_epochs=150, learning_rate=0.001, momentum=0.4, opt ='adam', activation='tanh'):
        """

        Creates many-to-many LSTM model by reframing time-series data as supervised learning X and Y data with shapes

        X.shape = (n_times, n_past, nx_features)
        Y.shape = (n_times, n_future, ny_features)

        where n_features is the number of columns in df. It then defines a model with the following architecture

        LSTM -> Dense Layer -> Reshape Layer

        such that input is of shape (n_past, n_features) and output is of shape (n_future, n_features). Could be extended in the future to select how many features to actually predict on.

        While the function has no output, it stores the model as a class variable, as well as the test/train X and Y data and model arguments.

        args:
            - n_past:           number of time points to use as input to LSTM
            - n_future:         number of time points to predict out of the Dense Layer
            - n_divide:         train/test split fraction
            - n_neurons:        number of neurons in LSTM layer
            - n_epochs:         number of epochs for training
            - learning_rate:    learning rate used in gradient descent
            - momentum:         momentum used in gradient descent
            - opt:              optimization routine
            - activation:       activation function in LSTM layer. No activation is implemented in Dense layer.
        """
        
        #calling normalized and scaled data
        x_data = self.x_data
        y_data = self.y_data
        nx_features = x_data.shape[1]
        ny_features = y_data.shape[1]
        self.nx_features = nx_features
        self.ny_features = ny_features
        
        # reframe as learning problem
        X, _ = self.PrepareData(x_data, n_past, n_future)
        _, Y = self.PrepareData(y_data, n_past, n_future)
        n_times = X.shape[0]
        n_divide = round((n_times)*n_divide)
        train_X, train_Y = X[:n_divide], Y[:n_divide]
        test_X, test_Y = X[n_divide:], Y[n_divide:]

        # keep track of Datetime index across train/test set
        self.train_index = self.index[:n_divide]
        self.test_index = self.index[n_divide:]
 
        #design network: n_neurons stands for size of hidden layer
        model = Sequential()
        model.add(LSTM(n_neurons, input_shape=(n_past, nx_features), activation=activation))
        model.add(Dense(n_future * ny_features))
        model.add(Reshape((n_future, ny_features)))
        if opt == 'sgd':
            Opt = optimizers.SGD(learning_rate = learning_rate,\
                                 momentum = momentum)
        if opt == 'adam':
            Opt = optimizers.Adam(learning_rate = learning_rate)
        model.compile(loss = 'mae', optimizer = Opt)

        # store
        self.opt = opt
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_past   = n_past
        self.n_future = n_future
        self.model    = model
        self.test_X   = test_X
        self.test_Y   = test_Y
        self.train_X  = train_X
        self.train_Y  = train_Y
        self.n_divide = n_divide

    def Fit(self):
        """
        Fits self.model and plots loss history over train and test sets.
        """

        model = self.model
        n_epochs = self.n_epochs
        train_X = self.train_X
        train_Y = self.train_Y
        test_X = self.test_X
        test_Y = self.test_Y

        # fit network
        print('Performing Training...')
        history = model.fit(train_X, train_Y, epochs=n_epochs, batch_size=24*14, validation_data=(test_X, test_Y), verbose=2, shuffle=False)
        print('...Training Done!')

        #plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.xlabel('epoch')
        plt.ylabel('loss [MSE]')
        plt.legend()
        plt.show()

    def Predict(self):
        """
        predit Yhat on test_X, unscale, store in a DataFrame self.df_pred. Currently just takes the predictions from max n_future.
        """

        model = self.model
        X = self.test_X
        Yhat = model.predict(X)
        self.Yhat = Yhat
        back = Yhat.shape[0]
        Yhat_last = Yhat[:,-1,:] # keep just the last time point predicted for each time
        index = self.index[-back:]
        columns = self.df_y.columns
        df_scaled = pd.DataFrame(Yhat_last, columns=columns, index=index) # not sure yet how to handle index
        df = self.invNormAndScale(df_scaled, self.y_scaler)
        self.df_yhat = df

        rmse = np.mean(np.sqrt((Yhat-self.test_Y)**2))
        print('Test RMSE: %.3f' % rmse)

    def Forecast(self):
        """
        Macro that predicts forecast on test_X and then plots.
        """
        self.Predict()
        self.PlotData([self.df_y, self.df_yhat])

    ### Helper Functions ###

    def PlotData(self, dfs=None, log_vars=['Eruption_Activity', 'VPCC_RSAM', 'VPPC_RSAM', 'VPNC_RSAM', 'VPRS_RSAM']):
        """
        Plots data for each DataFrame in dfs:

        args:
            - dfs:              (list) DataFrame objects to plot. Each DataFrame must have matching columns
            - log_vars:         (list) names of columns to plot in log scale
        """

        if dfs is None:
            dfs = [self.df]
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
        columns = dfs[0].columns
        scale = 3
        fig, ax = plt.subplots(math.ceil(len(columns)/2), 2, figsize=(8*scale, 0.75*scale*len(columns)), constrained_layout=True)
        ax = ax.flatten()

        for df in dfs:
            for ii, c in enumerate(columns):
                df[c].plot(ax=ax[ii])
                ax[ii].set(ylabel=c)
                if c in log_vars:
                    ax[ii].set_yscale('log')

        fig.tight_layout(pad=10)
        plt.show()

    def PrepareData(self, data, n_past, n_future):
        """
        Reframes time-series data as learning X and Y data.

            data.shape = (n_times, n_features)
            x.shape = (n_times*, n_past, n_features)
            y.shape = (n_times*, n_future, n_features)

        where n_times* = n_times - n_past - n_future + 1

        args:
            - data:             (ndarray)          
            - n_past:           (int)
            - n_future:         (int)

        return:
            - X:                (ndarray)
            - Y:                (ndarray)
        """
        X, Y = [], []
        for i in range(n_past, len(data) - n_future + 1):
            X.append(data[i - n_past:i])
            Y.append(data[i:i + n_future])
        return np.array(X), np.array(Y)
    
    def NormAndScale(self, df):
        """
        Takes DataFrame as input and returns another DataFrame with same shape with scaled data from [0,1] using MinMaxScaler

        args:
            - df:         (DataFrame) data to scale

        returns:
            - df_scaled   (DataFrame) scaled data
            - scaler:     (MinMaxScaler) scaler transform object
        """
        
        # extracting values from dataframe
        df_scaled = df.copy()
        data = df_scaled.to_numpy()
        
        # normalize
        scaler = MinMaxScaler(feature_range=(0, 1)) #normalize features
        scaled = scaler.fit_transform(data)
        
        # storing scaled values into a dataframe
        df_scaled[:] = scaled
            
        return df_scaled, scaler

    def invNormAndScale(self, df_scaled, scaler):
        """
        Inverse of NormAndScale. Takes as input scaled data and scaler and return unscaled data

        args:
            - df_scaled   (DataFrame) scaled data
            - scaler:     (MinMaxScaler) scaler transform object

        scaler:
            - df:         (DataFrame) unscaled data
        """

        # extracting values from dataframe
        df = df_scaled.copy()
        data = df.to_numpy()
        
        # undo normaliztion
        unscaled = scaler.inverse_transform(data)

        # store unscaled values and return
        df[:] = unscaled

        return df