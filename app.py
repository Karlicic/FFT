import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 8192

N = 128


def fft(x):
    X = list()
    for k in range(0, N):
        X.append(np.complex(x[k], 0))

    fft_recursion(X)
    return X


def fft_recursion(X):
    N = len(X)
    if N <= 1:
        return
    even = np.array(X[0:N:2])
    odd = np.array(X[1:N:2])

    fft_recursion(even)
    fft_recursion(odd)

    for k in range(0, int(N / 2)):
        t = np.exp(np.complex(0, -2 * np.pi * k / N)) * odd[k]
        X[k] = even[k] + t
        X[int(N / 2) + k] = even[k] - t


x_values = np.arange(0, N, 1)

x = np.sin((2 * np.pi * x_values / 32.0))  # 32 - 256Hz
x += np.sin((2 * np.pi * x_values / 64.0))  # 64 - 128Hz

X = fft(x)

# Plotting
_, plots = plt.subplots(2)

# Plot in time domain
plots[0].plot(x)

# Plot in frequent domain
powers_all = np.abs(np.divide(X, N / 2))
powers = powers_all[0:int(N / 2)]
frequencies = np.divide(np.multiply(SAMPLE_RATE, np.arange(0, N / 2)), N)
plots[1].plot(frequencies, powers)

# Show plots
plt.show()
