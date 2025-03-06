import numpy as np 
import pandas as pd
import math
from abc import ABC, abstractmethod
from math import log, sqrt, exp
from scipy.stats import norm



'''
Standardized inputs:
    - 
'''
class Forward_Method(ABC):
  @abstractmethod
  def get_option_prices():
    pass


class Black_Scholes(Forward_Method):
  def get_options_prices(self, S, K, r, T, sigma):
    return self.try_1(S, K, r, T, sigma)

  def try_1(S, K, r, T, sigma):
    """
    Black-Scholes formula for a European call option.
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

class Local_Black_Scholes(Forward_Method):
  def get_option_prices(self, S0, K, T, r, sigma_func,):
    return self.black_scholes_local_vol(S0, K, T, r, sigma_func,)

  ###### Claude's attempt: #######
  def black_scholes_local_vol(self, S0, T, r, local_vol, option_type='call', 
                           num_S_steps=100, num_t_steps=100):
    result = []
    for K in range(300):
      result.append(self.black_scholes_local_vol(S0, K, T, r, local_vol, option_type='call', num_S_steps=100, num_t_steps=100))
    return result
        

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
  
  ######### ChatGPT #########
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