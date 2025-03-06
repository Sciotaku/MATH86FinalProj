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
        return [self.black_scholes_fdm(K, sigma_func, option_type) 
                for K in strike_prices]

    def black_scholes_fdm(self, K, sigma_func, option_type="call", 
                          num_S_steps=100, num_t_steps=100):
        """
        A simple explicit FD scheme for demonstration 
        (not the most stable for large steps).
        """
        S_max = max(self.S0, K) * 3
        S_min = max(0.01, self.S0 / 3)
        S_array = np.linspace(S_min, S_max, num_S_steps)
        dt = self.T / num_t_steps

        # Final payoff
        if option_type == "call":
            V = np.maximum(S_array - K, 0)
        else:
            V = np.maximum(K - S_array, 0)

        dS = S_array[1] - S_array[0]

        for _ in range(num_t_steps):
            V_new = np.zeros_like(V)

            for i in range(1, num_S_steps - 1):
                S = S_array[i]
                sigma = sigma_func(S)
                
                # FD coefficients for an explicit scheme
                a = 0.5 * dt * (sigma**2 * S**2 / dS**2 - self.r * S / dS)
                b = 1 - dt * (sigma**2 * S**2 / dS**2 + self.r)
                c = 0.5 * dt * (sigma**2 * S**2 / dS**2 + self.r * S / dS)

                V_new[i] = a * V[i-1] + b * V[i] + c * V[i+1]

            # Boundary conditions
            if option_type == "call":
                V_new[0]  = 0  # call -> worthless at S=0
                V_new[-1] = S_array[-1] - K * np.exp(-self.r * self.T)
            else:
                V_new[0]  = K * np.exp(-self.r * self.T)  # put -> near S=0 payoff ~ K e^{-rT}
                V_new[-1] = 0

            V = V_new

        # Interpolate FD grid back to current underlying S0
        return np.interp(self.S0, S_array, V)

# -------------------------------
# Butterfly Class
# -------------------------------
class Butterfly:
    @staticmethod
    def get_density(strikes, prices, time_to_expiry, R, D):
        """
        Breeden-Litzenberger approach: 
            density ~ exp(R*T) * 2nd derivative wrt strike of CallPrice(K).
        For discrete strikes, approximate via butterfly spread.
        """
        strikes = np.asarray(strikes)
        prices = np.asarray(prices)

        if len(strikes) < 3 or len(prices) < 3:
            raise ValueError("Need at least 3 strike prices to compute butterfly spreads.")

        densities = np.zeros(len(strikes) - 2)
        for i in range(1, len(strikes) - 1):
            left_price, middle_price, right_price = prices[i - 1], prices[i], prices[i + 1]
            butterfly_price = left_price + right_price - 2.0 * middle_price

            strike_diff = strikes[i + 1] - strikes[i]
            if strike_diff > 0:
                density = np.exp(R * time_to_expiry) * butterfly_price / (strike_diff**2)
            else:
                density = 0
            densities[i - 1] = max(density, 0)

        return densities

# -------------------------------
# Local-Vol Fokker-Planck PDE
# -------------------------------
def fokker_planck_local(strike_prices, time_to_expiry, r, local_vol_func, 
                        num_steps=100):
    """
    Compute RND using a forward Fokker-Planck PDE in S-space
    with local volatility sigma = local_vol_func(S).
    
    PDE: ∂p/∂t = -∂(r S p)/∂S + 0.5 ∂²( (sigma(S) * S)² * p )/∂S²

    Parameters
    ----------
    strike_prices   : array of strike prices (used as the S-grid)
    time_to_expiry  : T
    r               : risk-free rate
    local_vol_func  : function sigma(S)
    num_steps       : number of time steps

    Returns
    -------
    S_vals, p       : final distribution p(S, t=T)
    """
    S_min, S_max = np.min(strike_prices), np.max(strike_prices)
    S_vals = np.linspace(S_min, S_max, len(strike_prices))
    dS = S_vals[1] - S_vals[0]
    dt = time_to_expiry / num_steps
    n = len(S_vals)

    # 1) Evaluate local vol at each grid point
    sigma_vals = np.array([local_vol_func(S) for S in S_vals])
    sigma_vals = np.clip(sigma_vals, 0.01, 1.0)  # safeguard

    # 2) Initial distribution: try a simple Gaussian around the midpoint
    p = np.exp(-0.5 * ((S_vals - np.median(S_vals))**2) / (0.2*S_vals.mean())**2)
    p /= np.sum(p * dS)

    # 3) Crank–Nicholson Coeffs
    alpha = 0.5 * r * S_vals / dS
    beta  = 0.5 * sigma_vals**2 * (S_vals**2) / (dS**2)

    # Build diagonals
    A_diag = -beta - alpha / 2
    B_diag = -beta + alpha / 2
    C_diag = 2*beta + 1.0/dt
    D_diag = -2*beta + 1.0/dt

    A = diags([A_diag[1:], C_diag, B_diag[:-1]],
              offsets=[-1, 0, 1], shape=(n, n), format="csr")
    B = diags([-A_diag[1:], D_diag, -B_diag[:-1]],
              offsets=[-1, 0, 1], shape=(n, n), format="csr")

    # 4) Boundary rows => p=0 at edges
    for mat in (A, B):
        mat[0, 0] = 1.0
        mat[0, 1] = 0.0
        mat[-1, -1] = 1.0
        mat[-1, -2] = 0.0

    # 5) Time stepping
    for _ in range(num_steps):
        rhs = B @ p
        # Enforce boundary on RHS
        rhs[0]  = 0.0
        rhs[-1] = 0.0

        p_new = spsolve(A, rhs)

        # Set boundary
        p_new[0]  = 0.0
        p_new[-1] = 0.0

        # Normalize
        mass = np.sum(p_new * dS)
        if mass > 1e-15:
            p_new /= mass

        p = p_new

    return S_vals, p

# -------------------------------
# Synthetic Volatility Functions
# -------------------------------
def constant_volatility(S):   return 0.2
def normal_volatility(S):     return np.abs(np.random.normal(loc=0.2, scale=0.05))
def poisson_volatility(S):    return np.abs((np.random.poisson(lam=3) / 30) + 0.1)
def exponential_volatility(S):return np.random.exponential(scale=0.1)

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    num_points = 100
    strike_prices = np.linspace(50, 150, num_points)

    S0, T, r = 100, 0.5, 0.05
    lbs = LocalBlackScholes(S0, T, r)

    # --- 1) Generate option prices with each local vol function
    option_prices = {
        "Constant Volatility": np.array(lbs.get_option_prices(strike_prices, constant_volatility)),
        "Normal Volatility":   np.array(lbs.get_option_prices(strike_prices, normal_volatility)),
        "Poisson Volatility":  np.array(lbs.get_option_prices(strike_prices, poisson_volatility)),
        "Exponential Volatility": np.array(lbs.get_option_prices(strike_prices, exponential_volatility)),
    }

    # --- 2) Construct local vol for PDE and solve for distribution
    #     We'll sample the same functions over [S_min..S_max], 
    #     build an interpolant, then pass to fokker_planck_local.
    def build_local_vol_func(vol_func, S_min, S_max, n_samples=10):
        S_grid = np.linspace(S_min, S_max, n_samples)
        samples = [vol_func(S) for S in S_grid]
        samples = np.clip(samples, 0.01, 1.0)
        return interp1d(S_grid, samples, kind="cubic", fill_value="extrapolate")

    # We'll do a "true_rnd" using the local vol PDE (instead of the old constant PDE)
    S_min, S_max = 50, 150
    true_rnd = {}
    for vol_type, vol_func in [
        ("Constant Volatility", constant_volatility),
        ("Normal Volatility",   normal_volatility),
        ("Poisson Volatility",  poisson_volatility),
        ("Exponential Volatility", exponential_volatility),
    ]:
        # Build local vol function
        local_vol = build_local_vol_func(vol_func, S_min, S_max)

        # Solve PDE
        S_vals, p = fokker_planck_local(strike_prices, T, r, local_vol, num_steps=50)
        true_rnd[vol_type] = p

    # --- 3) Compute RND from Butterfly Spreads
    butterfly_rnd = {}
    for vol_type in option_prices:
        bfly = Butterfly.get_density(strike_prices, option_prices[vol_type], T, r, D=0)
        butterfly_rnd[vol_type] = np.array(bfly)

    # --- 4) Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    vol_keys = list(option_prices.keys())
    ax_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for vol_type, ax_idx in zip(vol_keys, ax_indices):
        ax = axes[ax_idx]
        # "True" PDE-based RND (local vol)
        ax.plot(S_vals, true_rnd[vol_type], 
                label=f"FP PDE RND - {vol_type}", 
                linestyle="--", linewidth=2)

        # Butterfly-based RND
        # Notice butterfly_rnd has length = len(strikes)-2
        # We'll center it on the interior strikes
        ax.plot(strike_prices[1:-1], butterfly_rnd[vol_type], 
                label=f"Butterfly RND - {vol_type}", 
                color='red', linestyle="-", marker='o')

        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Risk-Neutral Density")
        ax.set_title(f"Local Vol PDE vs. Butterfly: {vol_type}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()