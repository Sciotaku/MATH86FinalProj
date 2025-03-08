import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from methods import *

def solve_fokker_planck_crank_nicolson(
    S0: float,
    r: float,      # constant interest rate parameter
    d: float,      # constant dividend yield parameter
    T: float,       # total time (in years)
    sigma,     # function: sigma(S) -> float
    NS: int = 400,  # number of spatial steps
    S_max_factor: float = 5.0,  # domain up to S_max_factor * S0
    NT: int = 400,  # number of time steps
    upwind: bool = True  # Currently, the drift is implemented using an upwind scheme (assumed mu>0)
):
    """
    Solve the 1D Fokker–Planck PDE:
    
        dp/dt = 0.5 * d^2/dS^2 [ sigma^2(S)*S^2 * p ] - d/dS [ mu*S * p ]
    
    for S in [0, S_max] and t in [0, T] using a Crank–Nicolson scheme.
    
    The initial condition is a discrete delta at S = S0 and Dirichlet boundary conditions
    (p(t,0)=0, p(t,S_max)=0) are enforced.
    
    Parameters
    ----------
    S0 : float
        Initial stock price.
    sigma : callable or array-like or pd.Series
        Either a function sigma(S) or a discrete series of local volatilities.
        If not callable, the code will interpolate the provided series.
    mu : float
        Constant drift.
    T : float
        Total time (years).
    NS : int, optional
        Number of spatial grid intervals (default: 400).
    S_max_factor : float, optional
        Domain is [0, S_max_factor * S0] (default: 5.0).
    NT : int, optional
        Number of time steps (default: 400).
    upwind : bool, optional
        Currently the drift term is discretized in an upwind manner (recommended for mu>0).
    
    Returns
    -------
    S_grid : np.ndarray
        The spatial grid (stock prices).
    p_final : np.ndarray
        The probability density p(T,S) evaluated on S_grid.
    """
    
    # Create an interpolation function for sigma if sigma is not callable.
    if callable(sigma):
        sigma_func = sigma
    else:
        # If input is array-like or a pd.Series, build an interpolation.
        if isinstance(sigma, pd.Series):
            S_vals = sigma.index.astype(float).values
            sigma_vals = sigma.values.astype(float)
        else:
            sigma_vals = np.array(sigma).astype(float)
            # Assume that the array covers [0, S_max] with length N_sigma.
            N_sigma = len(sigma_vals)
            S_vals = np.linspace(0.0, S_max_factor * S0, N_sigma)
        sigma_func = lambda S: np.interp(S, S_vals, sigma_vals)
    
    # Define spatial grid.
    S_max = S_max_factor * S0
    S_grid = np.linspace(0.0, S_max, NS + 1)
    dS = S_grid[1] - S_grid[0]
    
    # Time step and drift component
    dt = T / NT
    mu = r-d
    
    # Initialize probability density: use a discrete delta at S0.
    p_now = np.zeros(NS + 1)
    p_next = np.zeros(NS + 1)
    i0 = np.argmin(np.abs(S_grid - S0))
    p_now[i0] = 1.0 / dS  # delta approximated as 1/dS at the nearest grid point.
    
    # -------------------------------------------------------------------------
    # Build the spatial operator L such that dp/dt = L(p).
    # We write the PDE in flux form. For mu > 0 (upwind):
    #   flux_diff at i+1/2 = alpha_{i+1/2}*(p[i+1]-p[i]) / dS,
    #   flux_drift at i+1/2 = mu * S_{i+1/2} * p[i],
    # with alpha(S)=0.5*sigma(S)^2*S^2.
    #
    # The net flux is:
    #   flux_{i+1/2} = (alpha_{i+1/2}*(p[i+1]-p[i]) / dS) - mu * S_{i+1/2} * p[i].
    #
    # Then the finite-difference operator is:
    #   L(p)_i = (flux_{i+1/2} - flux_{i-1/2]) / dS,
    # which we represent as a tridiagonal matrix.
    # -------------------------------------------------------------------------
    
    # Compute alpha on the grid.
    alpha = np.array([0.5 * (sigma_func(S_grid[i])**2) * (S_grid[i]**2) for i in range(NS + 1)])
    # Compute midpoints alpha_{i+1/2} as simple averages.
    alpha_half = np.array([0.5 * (alpha[i] + alpha[i+1]) for i in range(NS)])
    
    # Set up tridiagonal coefficients: L(p)_i = a[i]*p[i-1] + b[i]*p[i] + c[i]*p[i+1].
    a = np.zeros(NS + 1)
    b = np.zeros(NS + 1)
    c = np.zeros(NS + 1)
    
    for i in range(1, NS):
        # Midpoints in S.
        S_half_right = 0.5 * (S_grid[i] + S_grid[i+1])
        S_half_left  = 0.5 * (S_grid[i] + S_grid[i-1])
        
        # At the right interface i+1/2:
        # Diffusive contribution:   alpha_half[i]*(p[i+1]-p[i])/dS,
        # Drift contribution (upwind):   mu * S_half_right * p[i].
        c_flux = alpha_half[i] / dS
        b_flux = - (alpha_half[i] / dS + mu * S_half_right)
        
        # At the left interface i-1/2:
        a_flux = - (alpha_half[i-1] / dS + mu * S_half_left)
        b_flux2 = alpha_half[i-1] / dS
        
        # Combine: L(p)_i = (flux_{i+1/2} - flux_{i-1/2})/dS.
        a[i] = -(a_flux) / dS      # multiplies p[i-1]
        b[i] = (b_flux - b_flux2) / dS  # multiplies p[i]
        c[i] = c_flux / dS         # multiplies p[i+1]
    
    # At the boundaries (i=0 and i=NS) we enforce Dirichlet: p=0.
    # (Thus a[0], c[0] and a[NS], c[NS] remain zero.)
    
    # -------------------------------------------------------------------------
    # Build the Crank–Nicolson matrices:
    #   M_minus = I - 0.5*dt*L  and  M_plus = I + 0.5*dt*L.
    # We represent these as tridiagonal matrices with arrays for lower, diag, upper.
    # -------------------------------------------------------------------------
    L_lower_m = np.zeros(NS + 1)
    L_diag_m  = np.zeros(NS + 1)
    L_upper_m = np.zeros(NS + 1)
    
    L_lower_p = np.zeros(NS + 1)
    L_diag_p  = np.zeros(NS + 1)
    L_upper_p = np.zeros(NS + 1)
    
    for i in range(NS + 1):
        if i == 0 or i == NS:
            L_diag_m[i] = 1.0
            L_diag_p[i] = 1.0
        else:
            L_diag_m[i] = 1.0 - 0.5 * dt * b[i]
            L_lower_m[i] = -0.5 * dt * a[i]
            L_upper_m[i] = -0.5 * dt * c[i]
            
            L_diag_p[i] = 1.0 + 0.5 * dt * b[i]
            L_lower_p[i] =  0.5 * dt * a[i]
            L_upper_p[i] =  0.5 * dt * c[i]
    
    # -------------------------------------------------------------------------
    # Define a tridiagonal solver that does not modify the original coefficients.
    # -------------------------------------------------------------------------
    def solve_tridiagonal(lsub, diag, lsup, rhs):
        n = len(rhs)
        diag_copy = diag.copy()
        rhs_copy = rhs.copy()
        for i in range(1, n):
            m = lsub[i] / diag_copy[i - 1]
            diag_copy[i] -= m * lsup[i - 1]
            rhs_copy[i] -= m * rhs_copy[i - 1]
        x = np.zeros(n)
        x[-1] = rhs_copy[-1] / diag_copy[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (rhs_copy[i] - lsup[i] * x[i + 1]) / diag_copy[i]
        return x
    
    # -------------------------------------------------------------------------
    # Time stepping: Crank–Nicolson scheme.
    # -------------------------------------------------------------------------
    for _ in range(NT):
        # Build right-hand side vector: rhs = M_plus * p_now.
        rhs = np.zeros(NS + 1)
        for i in range(NS + 1):
            if i == 0 or i == NS:
                rhs[i] = L_diag_p[i] * p_now[i]
            else:
                rhs[i] = (L_lower_p[i] * p_now[i - 1] +
                          L_diag_p[i]  * p_now[i] +
                          L_upper_p[i] * p_now[i + 1])
        # Solve for p_next from: M_minus * p_next = rhs.
        p_next = solve_tridiagonal(L_lower_m, L_diag_m, L_upper_m, rhs)
        # Correct for any small negative values.
        p_next[p_next < 0] = 0.0
        p_now = p_next.copy()
        # (Optionally, one may re-normalize to conserve mass.)
    
    return S_grid, p_now

# -------------------------------------------------------------------------
# Testing and visualization using a volatility series.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Set up parameters.
    S0 = 100.0       # initial stock price
    r = 0.05        # drift
    d = 0.02
    T = 1.0          # time horizon (1 year)
    NS = 400
    NT = 400
    S_max_factor = 5.0
    
    # Create a grid for S for the volatility series.
    S_grid_series = np.linspace(0.0, S_max_factor * S0, NS + 1)
    
    # Example 1: Local volatility that increases with sqrt(S).
    sigma_local = 0.20 + 0.1 * np.sqrt(S_grid_series / 100.0)
    sigma_local_series = pd.Series(sigma_local, index=S_grid_series)
    
    # Example 2: Constant volatility.
    sigma_const = 0.20 * np.ones_like(sigma_local)
    sigma_const_series = pd.Series(sigma_const, index=S_grid_series)
    
    # Solve using the series inputs.
    p_local = Forward_Method.fokker_planck(
        S0, r, d, T, sigma_local_series, NS, S_max_factor, NT
    )
    p_const = Forward_Method.fokker_planck(
        S0, r, d, T, sigma_const_series, NS, S_max_factor, NT
    )
    
    # Check the total probability (should be close to 1).
    # dS = S_grid_out[1] - S_grid_out[0]
    # total_prob_local = np.sum(p_local) * dS
    # total_prob_const = np.sum(p_const) * dS
    # print("Total probability (local vol series):", total_prob_local)
    # print("Total probability (constant vol series):", total_prob_const)
    
    # Plot the resulting PDFs.
    plt.figure(figsize=(10, 6))
    plt.plot(p_local, label="Local vol series: 0.20 + 0.1√(S/100)")
    plt.plot(p_const, label="Constant vol series: 0.20", linestyle="--")
    plt.xlabel("Stock Price S")
    plt.ylabel("Probability Density p(T,S)")
    plt.title("Fokker–Planck PDF at T = 1 year")
    plt.legend()
    plt.grid(True)
    plt.show()
