import numpy as np
# print(np.__doc__)
# print(help(np))
# print(dir(np))

def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    returns: numpy array or list of periodic portfolio returns (e.g. daily decimals)
    risk_free_rate: periodic risk-free rate, same frequency as returns
    """
    excess = np.array(returns) - risk_free_rate
    return np.mean(excess) / np.std(excess, ddof=1)

def batting_average(port_returns, bench_returns):
    """
    port_returns, bench_returns: arrays of equal length
    """
    port = np.array(port_returns)
    bench = np.array(bench_returns)

    wins = np.sum(port > bench) #port > bench returns a boolean array, and sum() counts how many are True.
    return wins / len(port)

def capture_ratios(port_returns, bench_returns):
    p = np.array(port_returns)
    b = np.array(bench_returns)
    # up-capture
    up_mask = b > 0  # similar to p[b>0] all element that satisfy condition
    upcap = p[up_mask].sum() / b[up_mask].sum() if up_mask.any() else np.nan #If all benchmark returns are ≤ 0, then .any() returns False.

    # down-capture
    down_mask = b < 0
    downcap = p[down_mask].sum() / b[down_mask].sum() if down_mask.any() else np.nan #else Not a Number
    return upcap, downcap


def tracking_error(port_returns, bench_returns):
    diff = np.array(port_returns) - np.array(bench_returns)
    return np.std(diff, ddof=1)


def max_drawdown(returns):
    """
    returns: array of periodic returns
    """
    returns = np.array(returns)
    # cumulative wealth index
    wealth = np.cumprod(1 + returns)     #Computes the cumulative product of array elements.
    peak = np.maximum.accumulate(wealth) #Returns the running maximum (i.e., max so far) of an array.
    drawdowns = (wealth - peak) / peak
    return drawdowns.min()
