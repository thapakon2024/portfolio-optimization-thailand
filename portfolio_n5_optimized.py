import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import itertools
import time
from typing import List, Tuple, Dict

# Enable memory growth for GPU (if available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)


@tf.function(reduce_retracing=True)
def calculate_portfolio_metrics_vectorized(P0, Wi, fund):
    """
    Ultra-optimized vectorized calculation for N=5
    All operations in TensorFlow for maximum speed
    
    Args:
        P0: Stock prices tensor [time_steps, n_stocks]
        Wi: Portfolio weights tensor [n_samples, n_stocks]
        fund: Investment fund amount (scalar)
    
    Returns:
        avg_return: Average portfolio return [n_samples]
        volatility: Portfolio volatility [n_samples]
        variance: Portfolio variance [n_samples]
    """
    # P0 shape: [time_steps, n_stocks]
    # Wi shape: [n_samples, n_stocks]
    
    # Amount of stocks for each portfolio
    # Broadcasting: Wi * fund / P0[0] -> [n_samples, n_stocks]
    X0 = Wi * fund / P0[0]
    
    # Portfolio value over time
    # matmul: [n_samples, n_stocks] × [n_stocks, time_steps] = [n_samples, time_steps]
    Port_Value = tf.matmul(tf.cast(X0, dtype=tf.float64), tf.transpose(P0))
    
    # Portfolio return (percentage)
    Port_return = (Port_Value - fund) / fund * 100.0
    
    # Average return across time
    avg_return = tf.reduce_mean(Port_return, axis=1)
    
    # Variance calculation
    variance = tf.math.reduce_variance(Port_return, axis=1)
    
    # Log returns (vectorized, avoid pandas shift)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    log_Port_Value = tf.math.log(Port_Value + epsilon)
    
    # Calculate differences using roll
    log_return = log_Port_Value - tf.roll(log_Port_Value, shift=1, axis=1)
    df_log = log_return - tf.roll(log_return, shift=2, axis=1)
    
    # Skip first 2 time steps (affected by roll)
    df_log_valid = df_log[:, 2:]
    
    # Sum of squared log differences
    sum_log = tf.reduce_sum(tf.square(df_log_valid), axis=1)
    
    # Volatility
    volatility = tf.sqrt(sum_log + epsilon)
    
    return avg_return, volatility, variance