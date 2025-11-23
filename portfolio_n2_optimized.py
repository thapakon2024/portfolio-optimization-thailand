import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf

# Function to calculate portfolio metrics

def calculate_portfolio_metrics(weights, expected_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Function to optimize portfolio

def optimize_portfolio(expected_returns, cov_matrix, target_return):
    num_assets = len(expected_returns)
    weights = tf.Variable(np.random.random(num_assets), dtype=tf.float32)
    weights = weights / tf.reduce_sum(weights)  # Normalize weights to sum to 1

    # Define the loss function (minimize volatility for target return)
    def loss_function():
        portfolio_return, portfolio_volatility = calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
        return portfolio_volatility if portfolio_return >= target_return else float('inf')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    for _ in range(1000):  # Optimization iterations
        optimizer.minimize(loss_function, var_list=[weights])

    return weights.numpy()

# Main function to download SET50 stock data and find optimal portfolio

def main():
    symbols = [...]
    df = yf.download(symbols, start='2022-01-01', end='2023-12-31')
    returns = df['Adj Close'].pct_change().dropna()
    expected_returns = returns.mean().values
    cov_matrix = returns.cov().values
    target_return = 0.01  # Target return of ~1%

    optimal_weights = optimize_portfolio(expected_returns, cov_matrix, target_return)
    print('Optimal Portfolio Weights:', optimal_weights)

if __name__ == '__main__':
    main()