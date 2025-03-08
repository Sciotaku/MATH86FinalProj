import numpy as np 
import pandas as pd
import math
from abc import ABC, abstractmethod
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


#--------------------------------------------------#
'''
(vol ---> option prices)
Forward Methods:

  get_option_price(
    Inputs:
      - S0: stock price (float)
      - K: strike price (float)
      - r: yearly interest rate (float)
          (5% interest --> R=0.05)
      - d: yearly dividend yield/rate (float)
          (2% yearly dividend yield --> d=0.02))
      - T: time to expiry (float)
          (in fraction of years, so 25 trading days ---> T=25/252)
      - sigma: volatility (float || pd.Series)
    ): --> 
    Output: options price (float)
  (Note: get_options_price only returns one options price for a given strike, thus to get a list of "market prices" at different strikes, the function must be called several times)

  fokker_plank(
    Inputs:
      - S0: initial stock price (float)
      - r: yearly interest rate (float)
          (5% interest --> R=0.05)
      - d: yearly dividend yield/rate (float)
          (2% yearly dividend yield --> d=0.02))
      - T: time to expiry (float)
          (in fraction of years, so 25 trading days ---> T=25/252)
      - sigma: local volatility (pd.Series)
          (function of price)
      - (optional) NS: number of spatial steps
      - (optional) S_max_factor: the max geometric increase of the stock in the calculated pdf
      - (optional) NT: number of time steps
      - (optional) upwind: If True, use an upwind discretization for the drift term (recommended when mu > 0)
  ): -->
  Output: 
'''
class Forward_Method(ABC):
  @abstractmethod
  def get_option_price(S0: float, K: float, r: float, d: float, T: float, sigma: float | pd.Series) -> float:
    pass
  
  # TODO: Impliment
  @abstractmethod
  def fokker_planck(
      S0: float,
      r: float,      # constant interest rate parameter
      d: float,      # constant dividend yield parameter
      T: float,       # total time (in years)
      sigma,         # function: sigma(S) -> float, or array-like/pd.Series of volatilities
      NS: int = 400,  # number of spatial steps
      S_max_factor: float = 5.0,  # domain up to S_max_factor * S0
      NT: int = 400,  # number of time steps
      upwind: bool = True  # Currently, the drift is implemented using an upwind scheme (assumed mu > 0)
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
      r : float
          Constant interest rate.
      d : float
          Constant dividend yield.
      T : float
          Total time (years).
      sigma : callable or array-like or pd.Series
          Either a function sigma(S) or a discrete series of local volatilities.
          If not callable, the code will interpolate the provided series.
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
      pd.Series
          A Pandas Series where the index is the S_grid and the values are the probability density p(T,S).
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
      
      # Time step and drift component.
      dt = T / NT
      mu = r - d  # risk-neutral drift
      
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
      
      # -------------------------------------------------------------------------
      # Build the Crank–Nicolson matrices:
      #   M_minus = I - 0.5*dt*L  and  M_plus = I + 0.5*dt*L.
      # Represent these as tridiagonal matrices with arrays for lower, diag, and upper.
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
          # (Optionally, re-normalize to conserve mass.)
      
      return pd.Series(index=S_grid, data=p_now)

# TODO: Double check if works
class Black_Scholes(Forward_Method):
  @staticmethod
  def get_option_price(S0: float, K: float, r: float, d: float, T: float, sigma: float | pd.Series) -> float:
     return Black_Scholes.try_1(S0, K, r, T, sigma)

  @staticmethod
  def try_1(S0, K, r, T, sigma):
    """
    Black-Scholes formula for a European call option.
    """
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S0 * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call
  
  def fokker_plank():
    return super.fokker_plank()

# TODO: Fix (idk if it works)
class Local_Black_Scholes(Forward_Method):
  def get_option_price(S0, K, r, d, T, sigma):
     return Local_Black_Scholes.claude_bs(S0, K, r, T, sigma)

  ###### Claude's attempt: #######
  @staticmethod
  def bs_loop(S0, T, r, local_vol, strike_range=[0,301], option_type='call', num_S_steps=100, num_t_steps=100):
    result = []
    for K in range(strike_range[0], strike_range[1]):
      result.append(Local_Black_Scholes.claude_bs(S0, K, T, r, local_vol, option_type, num_S_steps, num_t_steps))
    return result
        
  @staticmethod
  def claude_bs(S0, K, r, T, local_vol, option_type='call', 
                           num_S_steps=100, num_t_steps=100):
    """
    Black-Scholes with price-dependent volatility using finite difference method
    
    Parameters:
    S0 : float - Initial stock price
    K : float - Strike price
    T : float - Time to maturity (in years)
    r : float - Risk-free interest rate (annualized)
    local_vol : function - A function that takes stock price S as input and returns local volatility
                Example: lambda S: 0.2 * (1 + 0.1 * np.abs(np.log(S/K)))
    option_type : str - 'call' or 'put'
    num_S_steps : int - Number of stock price steps in the grid
    num_t_steps : int - Number of time steps
    
    Returns:
    float - Option price
    """
    # Set up the grid boundaries
    S_max = max(S0, K) * 3
    S_min = max(0.01, S0 / 3)
    
    # Create grid of stock prices and time steps
    S_array = np.linspace(S_min, S_max, num_S_steps)
    dt = T / num_t_steps
    
    # Initialize option values at maturity
    if option_type.lower() == 'call':
        V = np.maximum(S_array - K, 0)
    else:  # Put option
        V = np.maximum(K - S_array, 0)
    
    # Backward induction through time
    for t in range(num_t_steps):
        dS = S_array[1] - S_array[0]  # Grid spacing
        V_new = np.zeros_like(V)
        
        # Apply finite difference at each interior stock price point
        for i in range(1, num_S_steps - 1):
            S = S_array[i]
            sigma = local_vol(S)  # Get local volatility at current price
            
            # Finite difference coefficients for explicit scheme
            a = 0.5 * dt * (sigma**2 * S**2 / dS**2 - r * S / dS)
            b = 1 - dt * (sigma**2 * S**2 / dS**2 + r)
            c = 0.5 * dt * (sigma**2 * S**2 / dS**2 + r * S / dS)
            
            # Update option value
            V_new[i] = a * V[i-1] + b * V[i] + c * V[i+1]
        
        # Apply boundary conditions
        if option_type.lower() == 'call':
            V_new[0] = 0  # At S=0, call option is worthless
            V_new[-1] = S_array[-1] - K * np.exp(-r * (T - t*dt))  # At S→∞, call option ~ S-Ke^(-r*τ)
        else:  # Put option
            V_new[0] = K * np.exp(-r * (T - t*dt))  # At S=0, put option ~ Ke^(-r*τ)
            V_new[-1] = 0  # At S→∞, put option is worthless
        
        V = V_new
    
    # Interpolate to find the option price at S0
    option_price = np.interp(S0, S_array, V)
    
    return option_price
  
  ######### ChatGPT's attempt #########
  def chatgpt_price(S0, K, r, T, sigma_func, S_max=None, M=200, N=200, option_type="call"):
    """
    Price a European call or put option under the Black-Scholes model 
    with price-dependent volatility using a finite difference method (Crank-Nicolson).
    Returns a tuple: (option_price_at_S0, S_grid, option_values_at_t0_for_all_S).
    """
    # Choose a default S_max if not provided (e.g., 3x of max(S0,K))
    if S_max is None:
        S_max = 3 * max(S0, K)
    S_max = float(S_max)
    M = int(M); N = int(N)
    dS = S_max / M                        # price grid size
    dt = T / N                            # time step size
    S_grid = np.linspace(0, S_max, M+1)   # array of S values from 0 to S_max
    
    # Terminal payoff at t=T
    if option_type == "call":
        V_old = np.maximum(S_grid - K, 0.0)
    elif option_type == "put":
        V_old = np.maximum(K - S_grid, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Precompute finite difference coefficients for interior points
    a = np.zeros(M+1);  b = np.zeros(M+1);  c = np.zeros(M+1)
    for i in range(1, M):
        sigma_val = sigma_func(S_grid[i])
        a[i] = 0.5 * sigma_val**2 * S_grid[i]**2 / (dS**2) - 0.5 * r * S_grid[i] / dS
        c[i] = 0.5 * sigma_val**2 * S_grid[i]**2 / (dS**2) + 0.5 * r * S_grid[i] / dS
        b[i] = - (a[i] + c[i]) - r
    # Set up the tridiagonal matrix for (I - 0.5*dt*A)
    n = M - 1  # number of interior unknowns
    sub = np.zeros(n-1);  mid = np.zeros(n);  sup = np.zeros(n-1)
    for j in range(n):
        i = j + 1  # actual grid index
        mid[j] = 1 - 0.5 * dt * b[i]
        if j > 0:
            sub[j-1] = -0.5 * dt * a[i]    # sub-diagonal element (for V_{i-1})
        if j < n-1:
            sup[j]   = -0.5 * dt * c[i]    # super-diagonal element (for V_{i+1})
    # Perform LU factorization of the tridiagonal matrix (Thomas algorithm prep)
    L_fac = np.zeros(n-1); 
    # We will modify mid[] in place to store the diagonal of U in LU.
    for j in range(1, n):
        # Factor for sub-diagonal elimination
        L_fac[j-1] = sub[j-1] / mid[j-1]
        mid[j] = mid[j] - L_fac[j-1] * sup[j-1]

     # Time-marching backward from j=N (t=T) to j=0 (t=0)
    for step in range(N):
        t_curr = T - step * dt        # current time (starting from T downwards)
        t_new  = T - (step+1) * dt    # next time we are stepping to
        # Boundary values at the new time layer (t_new)
        if option_type == "call":
            V_new_left  = 0.0                                  # V(0, t) = 0
            V_new_right = S_max - K * math.exp(-r * (T - t_new))
        else:  # put option
            V_new_left  = K * math.exp(-r * (T - t_new))
            V_new_right = 0.0
        # Construct the RHS = (I + 0.5*dt*A) * V_old (for interior points 1..M-1)
        RHS = np.zeros(n)
        for j in range(n):
            i = j + 1  # actual grid index
            RHS[j] = V_old[i] + 0.5 * dt * (a[i] * V_old[i-1] + b[i] * V_old[i] + c[i] * V_old[i+1])
        # Adjust RHS for boundary contributions from V_new[0] and V_new[M]
        RHS[0]    += 0.5 * dt * a[1]    * V_new_left   # contribution at i=1 from S=0 boundary
        RHS[n-1] += 0.5 * dt * c[M-1]  * V_new_right  # contribution at i=M-1 from S=S_max boundary
        # Solve tridiagonal system (LU solve) for interior V_new values:
        # Forward substitution (apply L factors to RHS)
        for j in range(1, n):
            RHS[j] -= L_fac[j-1] * RHS[j-1]
        # Back substitution (solve U * x = RHS)
        V_new_int = np.zeros(n)
        if n > 0:
            V_new_int[n-1] = RHS[n-1] / mid[n-1]
        for j in range(n-2, -1, -1):
            V_new_int[j] = (RHS[j] - sup[j] * V_new_int[j+1]) / mid[j]
        # Combine interior and boundaries into the full price array
        V_new = np.empty(M+1)
        V_new[0]  = V_new_left
        V_new[1:M] = V_new_int
        V_new[M]  = V_new_right
        # Move to next time step (update V_old for next iteration)
        V_old = V_new

      # After time-stepping, V_old contains option values at t=0 for all S in grid
    V_t0 = V_old  
    # Interpolate to get value at S0
    if S0 <= 0:
        price_at_S0 = V_t0[0]
    elif S0 >= S_max:
        price_at_S0 = V_t0[-1]
    else:
        # find indices around S0
        idx = int(S0 / dS)
        # Linear interpolation between grid[idx] and grid[idx+1]
        S_left, S_right = S_grid[idx], S_grid[idx+1]
        V_left, V_right = V_t0[idx], V_t0[idx+1]
        price_at_S0 = V_left + (V_right - V_left) * (S0 - S_left) / (S_right - S_left)
    return price_at_S0, S_grid, V_t0
  
  ########### Deepseek's Attempt: ###########
  def deepseek_algo(S0, T, r, sigma_func, strike_range=[0,301],):
    result = []
    for K in range(strike_range[0], strike_range[1]):
      result.append(Local_Black_Scholes.deepseek_bs(S0, K, T, r, sigma_func,))
    return result

  def thomas_algorithm(a, b, c, d):
    """
    Solves a tridiagonal system using the Thomas algorithm.
    a: lower diagonal (length n-1)
    b: main diagonal (length n)
    c: upper diagonal (length n-1)
    d: right-hand side (length n)
    Returns x: solution vector (length n)
    """
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    
    # Forward sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
    
    # Last row
    denom = b[-1] - a[-1] * c_prime[-1]
    d_prime[-1] = (d[-1] - a[-1] * d_prime[-1]) / denom
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x

  def deepseek_bs(S0, K, T, r, sigma_func, Ns=200, Nt=1000, Smax=None, option_type='call'):
      if Smax is None:
          Smax = 2 * K
      Smin = 0
      dt = T / Nt
      ds = (Smax - Smin) / (Ns - 1)
      S_grid = np.linspace(Smin, Smax, Ns)
      
      # Precompute volatility values
      sigma_values = sigma_func(S_grid)
      
      # Initialize option price
      V = np.maximum(S_grid - K, 0) if option_type == 'call' else np.maximum(K - S_grid, 0)
      
      # Coefficients for interior points (j=1 to Ns-2)
      a = np.zeros(Ns)
      b = np.zeros(Ns)
      c = np.zeros(Ns)
      
      for j in range(1, Ns-1):
          S_j = S_grid[j]
          sigma_j = sigma_values[j]
          a[j] = 0.5 * sigma_j**2 * S_j**2 / ds**2 - 0.5 * r * S_j / ds
          b[j] = -sigma_j**2 * S_j**2 / ds**2 - r
          c[j] = 0.5 * sigma_j**2 * S_j**2 / ds**2 + 0.5 * r * S_j / ds
      
      # Time stepping
      for m in range(Nt):
          current_time = m * dt
          tau = T - current_time
          
          # Boundary conditions
          if option_type == 'call':
              V[-1] = Smax - K * np.exp(-r * tau)
              V[0] = 0.0
          else:
              V[0] = K * np.exp(-r * tau)
              V[-1] = 0.0
          
          # Tridiagonal coefficients
          main_diag = 1 - 0.5 * dt * b[1:-1]
          lower_diag = -0.5 * dt * a[2:-1]
          upper_diag = -0.5 * dt * c[1:-2]
          
          # Construct RHS
          rhs = np.zeros(Ns-2)
          for j in range(1, Ns-1):
              rhs[j-1] = V[j] + 0.5 * dt * (a[j] * V[j-1] + b[j] * V[j] + c[j] * V[j+1])
          
          # Adjust RHS for boundary terms
          rhs[0] += 0.5 * dt * a[1] * V[0]
          rhs[-1] += 0.5 * dt * c[Ns-2] * V[-1]
          
          # Solve tridiagonal system
          V_interior = Local_Black_Scholes.thomas_algorithm(lower_diag, main_diag, upper_diag, rhs)
          V[1:-1] = V_interior
      
      # Interpolate to S0
      option_price = interp1d(S_grid, V, kind='linear')(S0)
      return option_price

# TODO: Impliment
class Heston(Forward_Method):
  def get_option_price(S0, K, r, d, T, sigma):
     return super().get_option_price(K, r, d, T, sigma)


#--------------------------------------------------#
'''
(option prices ---> rnd)
Backward Methods:

  get_density(
    - option_prices: prices of options at different strikes (pd.Series)
  )
'''
class Backward_Method(ABC):
  @abstractmethod
  def get_density(option_prices: pd.Series) -> pd.Series:
    pass

# TODO: Impliment
class Isakov(Backward_Method):
  def get_density(option_prices):
     return super().get_density()

class Butterfly(Backward_Method):
  def get_density(option_prices: pd.Series, time_to_expiry, R, D,) -> pd.Series:
    option_prices = option_prices.sort_index()
    strikes = list(option_prices.index)
    prices = list(option_prices.values)
    butterfly_prices = {}
    for i in range(1,len(strikes)-1):
      # Get components of the butterfly spread
      strike = strikes[i]
      if (strikes[i-1] == strike - 1) and (strikes[i+1] == strike + 1):
        # print((strikes[i-1], strikes[i], strikes[i+1]))
        left = prices[i-1]
        middle = prices[i]
        right = prices[i+1]

        butterfly_price = left + right - (2 * middle)
        butterfly_prices[strike] = butterfly_price
    # TODO: Set to a pd.Series?
    return pd.Series(butterfly_prices)
  
  def butterfly_2(option_prices: pd.Series) -> pd.Series:
    option_prices = option_prices.sort_index()
    strikes = list(option_prices.index)
    prices = list(option_prices.values)
    butterfly_prices = {}
    for i in range(1, len(strikes) - 1):
        butterfly_prices[strikes[i]] = prices[i - 1] + prices[i + 1] - (2 * prices[i])
    return butterfly_prices

# TODO: Fix output to pd.Series   
class Breeden_Litzenberger(Backward_Method):
  def get_density(option_prices: pd.Series, time_to_expiry, R, D=0,) -> pd.Series:
    option_prices = option_prices.sort_index()
    strikes = list(option_prices.index)
    prices = list(option_prices.values)
    tau = time_to_expiry     # Time to maturity (years)
    r = R - D                # Annual risk-free rate - the dividend yeild

    h = strikes[1] - strikes[0]  # Step size for finite differences
    prices = pd.DataFrame({"strike": strikes, "price": prices})

    # Second derivative approximation using central finite differences
    prices['curvature'] = (-2 * prices['price'] + prices['price'].shift(1) + prices['price'].shift(-1)) / h**2

    # Apply the Breeden-Litzenberger formula
    prices['risk_neutral_pdf'] = np.exp(r * tau) * prices['curvature']

    # TODO: Set a standardized return thing: dictionary, pandas df?
    return prices
  
  def try_2(option_prices: pd.Series, time_to_expiry, R, D=0,) -> pd.Series:
    """
    Compute risk-neutral density using the second derivative of option prices
    according to Breeden-Litzenberger formula.
    """
    option_prices = option_prices.sort_index()
    strikes = list(option_prices.index)
    prices = list(option_prices.values)
    T = time_to_expiry     # Time to maturity (years)
    r = R - D                # Annual risk-free rate - the dividend yeild
    d2C_dK2 = np.gradient(np.gradient(prices, strikes), strikes)
    return np.exp(r * T) * d2C_dK2

# TODO: Impliment
class Implied_Vol(Backward_Method):
  def get_density(strikes, prices, time_to_expiry, R, D,):
    return super().get_density(prices)