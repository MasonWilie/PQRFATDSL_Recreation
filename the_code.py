import os
import datetime as dt
import random

from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

cwd = os.path.abspath('') + '/'
random.seed(1)
tf.random.set_seed(1)

def GARCH(y, start, end, split_date):
    """ General Autoregressive Conditional Heteroskedasticity Model 
    p -> Lag length
    q -> Order of ARCH terms
    y -> Timeseries Dataframe
    """
    model = arch_model(y, vol='Garch', p=1, o=0, q=1, dist='StudentsT')
    results = model.fit(last_obs=split_date)
    print(results.summary())

    sims = results.forecast(horizon=8, start=split_date)

    sims.variance[split_date:].plot()
    plt.title("Variance Forecast from " + split_date.strftime("%m/%d/%Y") +
              " to " + end.strftime("%m/%d/%Y") + " (m/d/y)")
    plt.ylabel("Variance [%]")
    plt.show()


def htqf_fun(p, tau):
    """Uses tensorflow functions to calculate the value at the quantile specified by tau

    This implements the HTQF (Heavy-Tailed Quantile Function) from Equation 6 in the paper:
    Q(τ|µ, σ, u, v) = µ + σ*Z_τ * (exp(u*Z_τ)/A + 1) * (exp(-v*Z_τ)/A + 1)
    where Z_τ is the τ-quantile of the standard normal distribution.

    Arguments:
        p {tensor} -- tensor that holds the mean, standard deviation, u-value (left tail weight), and v-value (right tail weight)
        tau {float or tensor} -- quantile level (e.g., 0.5 for median)

    Returns:
        tensor -- value at quantile tau
    """
    # Z_tau: τ-quantile of standard normal N(0,1)
    z_tau = tfp.distributions.Normal(loc=0.0, scale=1.0).quantile(tau)

    # HTQF formula from Equation 6
    return p[0] + tf.abs(p[1]) * z_tau * \
        (tf.math.exp(p[2] * z_tau) / 4 + 1) * \
        (tf.math.exp(-p[3] * z_tau) / 4 + 1)


def pinball_loss(y_true, y_pred):
    """Custom loss function for quantile regression over multiple quantiles

    This implements Equation 9 from the paper, which minimizes the sum of
    pinball losses over all K quantiles simultaneously:

    Loss = (1/K) * (1/(T-L)) * Σ_k Σ_t L_τk(r_t, Q(τ_k|µ_t, σ_t, u_t, v_t))

    where L_τ is the pinball loss from Equation 3:
    L_τ(y, q) = max(τ*(y-q), (τ-1)*(y-q))

    Arguments:
        y_true {tensor} -- True values (batch_size,)
        y_pred {tensor} -- Predicted HTQF parameters (batch_size, 4)
                          [mean, std_dev, u-value, v-value]

    Returns:
        tensor -- average pinball loss across all quantiles
    """
    # Define the 21 quantiles as in the paper
    K_quantiles = 21
    quantiles = [i / (K_quantiles + 1) for i in range(1, K_quantiles + 1)]

    total_loss = 0.0

    # Sum pinball losses over all K quantiles (Equation 9)
    for tau in quantiles:
        # Compute HTQF at quantile level tau for all samples in batch
        quant = tf.map_fn(
            fn=lambda p: htqf_fun(p, tau),
            elems=y_pred,
            dtype=tf.float32
        )

        # Calculate error: y_true - predicted_quantile
        err = y_true - quant

        # Pinball loss (Equation 3): max(τ*err, (τ-1)*err)
        # When y > q (err > 0): loss = τ * err
        # When y ≤ q (err ≤ 0): loss = (1-τ) * |err|
        loss_at_tau = tf.maximum(tau * err, (tau - 1) * err)

        # Accumulate loss for this quantile
        total_loss += tf.reduce_mean(loss_at_tau)

    # Average over all K quantiles
    return total_loss / K_quantiles


def moment(x, mu, order):
    """ Returns the moment sepecified by the order
    x       -> value
    mu      -> mean
    order   -> what moment to return

    Example:
        moment(x, mu, 2) -> second moment
        moment(x, mu, 3) -> third moment
    """
    if order == 1:
        return x

    return (x-mu)**order


def calculate_feature_vector(r_t, L):
    """ Calculates the feature vector for the LSTM Machine
    r_t -> returns
    L -> Lookback length
    """
    mu = sum(r_t[-L-1:-1]) / L

    return np.array([moment(r_t[-L], mu, 1),
                     moment(r_t[-L], mu, 2),
                     moment(r_t[-L], mu, 3),
                     moment(r_t[-L], mu, 4)])


def partition_data(ts, L, p_train, p_test, p_val):
    """Partition the data for training, testing, and validation

    Arguments:
        ts {np.array} -- Timeseries
        L {int} -- Lookback length
        p_train {float} -- Percentage of data to make training
        p_test {float} -- percentage of data to make testing
        p_val {float} -- percentage of data to make validation

    Returns:
        [type] -- [description]
    """
    x = []
    y = []

    for i in range(L+1, len(ts)):
        x.append(calculate_feature_vector(ts[i-L-1:i], L))
        y.append(ts[i])

    x = np.array(x)
    y = np.array(y)

    x_train = x[0:int(len(x)*p_train)]
    y_train = y[0:int(len(y)*p_train)]

    x_val = x[len(x_train):len(x_train) + int(len(x)*p_val)]
    y_val = y[len(x_train):len(x_train) + int(len(x)*p_val)]

    x_test = x[len(x_train) + len(x_val):len(x_train) +
               len(x_val)+int(len(x)*p_test)]
    y_test = y[len(x_train) + len(x_val):len(x_train) +
               len(x_val)+int(len(x)*p_test)]

    return np.expand_dims(x_train, axis=1), y_train, np.expand_dims(x_test, axis=1), y_test, np.expand_dims(x_val, axis=1), y_val


def main():
    start = dt.datetime(1970, 1, 1)
    end = dt.datetime(2018, 6, 1)
    data = yf.download('^GSPC', start=start, end=end)
    # Flatten column names if multi-index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    r_t = 100*data['Close'].pct_change().dropna()
    r_t = r_t.astype(float)
    plt.title("S&P500 Daily Returns")
    plt.ylabel("Change [%]")
    r_t.plot()
    plt.show()

    split_date = dt.datetime(2010, 1, 1)
    GARCH(r_t, start, end, split_date)

    # Plotting the actual variances (assuming mean = 0)
    actual_variance = r_t[split_date:].copy()
    mean_val = actual_variance.mean()
    actual_variance = actual_variance.apply(lambda x: np.abs(x) - mean_val)
    plt.figure()
    actual_variance.plot()
    plt.title("Real Variance from " + split_date.strftime("%m/%d/%Y") + " to " + end.strftime("%m/%d/%Y") + " (m/d/y)")
    plt.show()

    L_set = [40, 60, 80, 100]
    H_set = [8, 16]
    L = L_set[3]
    H = H_set[1]
    K_quantiles = 21
    quantiles = [i / (K_quantiles+1) for i in range(1, K_quantiles+1)]

    x_train, y_train, x_test, y_test, x_val, y_val = partition_data(r_t, L, 0.8, 0.1, 0.1)


    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(H, input_shape=(1, 4)))
    model.add(keras.layers.Dense(4, activation='tanh'))  # tanh as per Equation 8 in the paper

    print(model.summary())

    model.compile(loss=pinball_loss, optimizer='adam')
    model.fit(x_train, y_train, batch_size=1, validation_data=(x_val, y_val))

    h_t = model.predict(x_test)

    fig, axs = plt.subplots(2, 2)

    axs[0][0].plot(np.abs(h_t.T[0]))
    axs[0][0].set_title("Mean")

    axs[0][1].plot(np.abs(h_t.T[1]))
    axs[0][1].set_title("Standard Deviations")

    axs[1][0].plot(np.abs(h_t.T[2]))
    axs[1][0].set_title("u")

    axs[1][1].plot(np.abs(h_t.T[3]))
    axs[1][1].set_title("v")

    plt.show()


if __name__ == '__main__':
    main()
