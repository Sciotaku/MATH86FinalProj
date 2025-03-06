import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# -------------------------------
# Local Black-Scholes Class
# -------------------------------

class LocalBlackScholes:
    def __init__(self, S0, T, r):
        self.S0 = S0  
        self.T = T  
        self.r = r  

    def get_option_prices(self, strike_prices, sigma_func, option_type="call"):
        return [self.black_scholes_fdm(K, sigma_func, option_type) for K in strike_prices]

    def black_scholes_fdm(self, K, sigma_func, option_type="call", num_S_steps=100, num_t_steps=100):
        S_max = max(self.S0, K) * 3
        S_min = max(0.01, self.S0 / 3)
        S_array = np.linspace(S_min, S_max, num_S_steps)
        dt = self.T / num_t_steps

        V = np.maximum(S_array - K, 0) if option_type == "call" else np.maximum(K - S_array, 0)

        for _ in range(num_t_steps):
            dS = S_array[1] - S_array[0]
            V_new = np.zeros_like(V)

            for i in range(1, num_S_steps - 1):
                S = S_array[i]
                sigma = sigma_func(S)
                
                a = 0.5 * dt * (sigma**2 * S**2 / dS**2 - self.r * S / dS)
                b = 1 - dt * (sigma**2 * S**2 / dS**2 + self.r)
                c = 0.5 * dt * (sigma**2 * S**2 / dS**2 + self.r * S / dS)

                V_new[i] = a * V[i-1] + b * V[i] + c * V[i+1]

            V_new[0] = 0 if option_type == "call" else K * np.exp(-self.r * self.T)
            V_new[-1] = S_array[-1] - K * np.exp(-self.r * self.T) if option_type == "call" else 0

            V = V_new

        return np.interp(self.S0, S_array, V)

# -------------------------------
# Butterfly Class
# -------------------------------

class Butterfly:
    @staticmethod
    def get_density(strikes, prices, time_to_expiry, R, D):
        strikes = np.asarray(strikes)
        prices = np.asarray(prices)

        if len(strikes) < 3 or len(prices) < 3:
            raise ValueError("Need at least 3 strike prices to compute butterfly spreads.")

        densities = np.zeros(len(strikes) - 2)

        for i in range(1, len(strikes) - 1):
            left_price, middle_price, right_price = prices[i - 1], prices[i], prices[i + 1]
            butterfly_price = left_price + right_price - 2 * middle_price

            strike_diff = (strikes[i + 1] - strikes[i])
            density = np.exp(R * time_to_expiry) * butterfly_price / (strike_diff ** 2) if strike_diff > 0 else 0

            densities[i - 1] = max(density, 0)

        return densities

# -------------------------------
# Fokker-Planck PDE Solver (Improved)
# -------------------------------

def fokker_planck(strike_prices, option_prices, time_to_expiry, r, sigma, num_steps=100):
    """
    Compute risk-neutral density using an improved Fokker-Planck solver with Crank-Nicholson discretization.

    Parameters:
    - strike_prices: Array of strike prices (S)
    - option_prices: Array of option prices corresponding to the strikes
    - time_to_expiry: Time to expiration (T)
    - r: Risk-free rate
    - sigma: Constant volatility
    - num_steps: Number of time steps in the discretization

    Returns:
    - risk-neutral density as an array
    """

    # Convert to log-space for better numerical behavior
    S_min, S_max = np.min(strike_prices), np.max(strike_prices)
    S_vals = np.linspace(S_min, S_max, len(strike_prices))
    dS = S_vals[1] - S_vals[0]

    dt = time_to_expiry / num_steps

    # Initialize probability distribution as a log-normal approximation
    p = np.exp(-((S_vals - np.median(S_vals)) ** 2) / (2 * (sigma ** 2 * time_to_expiry)))
    p /= np.sum(p * dS)  # Normalize

    # Compute Crank-Nicholson coefficients
    sigma_vals = np.full(len(S_vals), sigma)  # Constant volatility
    alpha = (r / 2) * S_vals / dS  # Drift term
    beta = (sigma_vals**2 * S_vals**2) / (2 * dS**2)  # Diffusion term

    A_diag = -beta - alpha / 2
    B_diag = -beta + alpha / 2
    C_diag = 2 * beta + 1 / dt
    D_diag = -2 * beta + 1 / dt

    A = diags([A_diag[1:], C_diag, B_diag[:-1]], [-1, 0, 1], format="csr")
    B = diags([-A_diag[1:], D_diag, -B_diag[:-1]], [-1, 0, 1], format="csr")

    # Time stepping
    for _ in range(num_steps):
        b = B @ p
        p = spsolve(A, b)

    # Normalize final probability distribution
    p /= np.sum(p * dS)

    return p

# -------------------------------
# Main Execution
# -------------------------------

num_points = 100
np.random.seed(42)

strike_prices = np.linspace(50, 150, num_points)

def constant_volatility(S): return 0.2
def normal_volatility(S): return np.abs(np.random.normal(loc=0.2, scale=0.05))
def poisson_volatility(S): return np.abs((np.random.poisson(lam=3) / 30) + 0.1)
def exponential_volatility(S): return np.random.exponential(scale=0.1)

S0, T, r = 100, 0.5, 0.05
lbs = LocalBlackScholes(S0, T, r)

option_prices = {
    "Constant Volatility": np.array(lbs.get_option_prices(strike_prices, constant_volatility)),
    "Normal Volatility": np.array(lbs.get_option_prices(strike_prices, normal_volatility)),
    "Poisson Volatility": np.array(lbs.get_option_prices(strike_prices, poisson_volatility)),
    "Exponential Volatility": np.array(lbs.get_option_prices(strike_prices, exponential_volatility)),
}

true_rnd = {
    vol_type: fokker_planck(strike_prices, option_prices[vol_type], T, r, 0.2)
    for vol_type in option_prices
}

butterfly_rnd = {
    vol_type: Butterfly.get_density(strike_prices, option_prices[vol_type], T, r, D=0)
    for vol_type in option_prices
}

butterfly_rnd = {k: np.array(v) for k, v in butterfly_rnd.items()}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
volatility_types = list(option_prices.keys())
ax_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

for vol_type, ax_idx in zip(volatility_types, ax_indices):
    ax = axes[ax_idx]
    ax.plot(strike_prices, true_rnd[vol_type], label=f"True RND - {vol_type}", linestyle="--", linewidth=2)
    ax.plot(strike_prices[1:-1], butterfly_rnd[vol_type], label=f"Butterfly RND - {vol_type}", color='red', linestyle="-", marker='o')

    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Risk-Neutral Density")
    ax.set_title(f"True vs. Butterfly RND: {vol_type}")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
