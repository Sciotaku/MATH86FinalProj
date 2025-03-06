import numpy as np 
import pandas as pd
import math
from abc import ABC, abstractmethod
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def fokker_planck(vol_arr, time_to_expiry, r, num_steps=50, ):
  K_vals = vol_arr[0, :]  # Extract the first row from the 2D array vol_arr
  sigma_K = vol_arr[1, :] # Extract the 2nd row from the 2D array vol_arr
  # Fix negative values before interpolation
  sigma_K = np.clip(sigma_K, 0)
  # Interpolate sigma(K) to be used for all asset prices S
  # TODO: Check if this is the right way to do it
  local_vol_func = interp1d(K_vals, sigma_K, kind="cubic", fill_value="extrapolate")

  # 2. Define Fokker–Planck Parameters
  S_min, S_max = min(K_vals) // 25 , max(K_vals) * 25  # Asset price range
  N_S = 100  # Number of asset price grid points
  S_vals = np.linspace(S_min, S_max, N_S)  # Discretized price grid
  dS = S_vals[1] - S_vals[0]  # Step size in asset price space

  T = time_to_expiry
  N_T = num_steps  # Number of time steps
  dt = T / N_T  # Time step size

  # 3. Construct Crank–Nicholson Matrices
  sigma_vals = local_vol_func(S_vals)  # Compute sigma(S) at each grid point
  # Ensure all vol values remain above zero
  sigma_vals = np.clip(sigma_vals, 0)

  alpha = (r / 2) * S_vals / dS  # Drift term
  beta = (sigma_vals**2 * S_vals**2) / (2 * dS**2)  # Diffusion term

  A_diag = -beta - alpha / 2
  B_diag = -beta + alpha / 2
  C_diag = 2 * beta + 1 / dt
  D_diag = -2 * beta + 1 / dt

  # Create sparse matrices
  A = diags([A_diag[1:], C_diag, B_diag[:-1]], [-1, 0, 1], format="csr")  # Left matrix
  B = diags([-A_diag[1:], D_diag, -B_diag[:-1]], [-1, 0, 1], format="csr")  # Right matrix

  # 4. Initialize Probability Distribution
  pdf = np.exp(-((S_vals - 4050) ** 2) / (2 * 200 ** 2))  # Gaussian initial guess
  pdf /= np.sum(pdf * dS)  # Normalize to ensure total probability is 1

  # 5. Solve the Fokker–Planck PDE Using Crank–Nicholson
  for _ in range(N_T):
      b = B @ pdf  # Compute right-hand side
      pdf = spsolve(A, b)  # Solve linear system

  # Normalize final probability distribution
  pdf /= np.sum(pdf * dS)


class Forward_Method(ABC):
  @abstractmethod
  def get_option_prices():
    pass


class Black_Scholes(Forward_Method):
  @staticmethod
  def get_options_prices(S, K, r, T, sigma):
    return Black_Scholes.try_1(S, K, r, T, sigma)

  @staticmethod
  def try_1(S, K, r, T, sigma):
    """
    Black-Scholes formula for a European call option.
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

class Local_Black_Scholes(Forward_Method):
  @staticmethod
  def get_option_prices(S0, T, r, sigma_func, strike_range=[0,301],):
    return Local_Black_Scholes.bs_loop(S0, T, r, sigma_func, strike_range,)

  ###### Claude's attempt: #######
  @staticmethod
  def bs_loop(S0, T, r, local_vol, strike_range=[0,301], option_type='call', num_S_steps=100, num_t_steps=100):
    result = []
    for K in range(strike_range[0], strike_range[1]):
      result.append(Local_Black_Scholes.black_scholes_local_vol(S0, K, T, r, local_vol, option_type, num_S_steps, num_t_steps))
    return result
        
  @staticmethod
  def black_scholes_local_vol(S0, K, T, r, local_vol, option_type='call', 
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
  def chatgpt_price(S0, K, T, r, sigma_func, S_max=None, M=200, N=200, option_type="call"):
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
      result.append(Local_Black_Scholes.black_scholes_price_dependent_vol(S0, K, T, r, sigma_func,))
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

  def black_scholes_price_dependent_vol(S0, K, T, r, sigma_func, Ns=200, Nt=1000, Smax=None, option_type='call'):
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

class Heston(Forward_Method):
  def get_option_prices():
    pass

class Backward_Methods(ABC):
  @abstractmethod
  def get_density(strikes, prices):
    pass

class Isakov(Backward_Methods):
  def get_density(strikes, prices, time_to_expiry, R, D,):
    pass

class Butterfly(Backward_Methods):
  def get_density(strikes, prices, time_to_expiry, R, D,):
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
    # TODO: Set a standardized return thing: dictionary, pandas df?
    return butterfly_prices
  
  def butterfly_2(strikes, prices):
    butterfly_prices = {}
    for i in range(1, len(strikes) - 1):
        butterfly_prices[strikes[i]] = prices[i - 1] + prices[i + 1] - (2 * prices[i])
    return butterfly_prices
    
class Breeden_Litzenberger(Backward_Methods):
  def get_density(strikes, prices, time_to_expiry, R, D=0,):
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
  
  def try_2(strikes, prices, time_to_expiry, R, D=0,):
    """
    Compute risk-neutral density using the second derivative of option prices
    according to Breeden-Litzenberger formula.
    """
    T = time_to_expiry     # Time to maturity (years)
    r = R - D                # Annual risk-free rate - the dividend yeild
    d2C_dK2 = np.gradient(np.gradient(prices, strikes), strikes)
    return np.exp(r * T) * d2C_dK2

class Implied_Vol(Backward_Methods):
  def get_density(strikes, prices, time_to_expiry, R, D,):
    return super().get_density(prices)