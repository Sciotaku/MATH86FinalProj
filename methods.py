import numpy as np 
import pandas as pd
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
  def get_option_prices():
    pass

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

class Implied_Vol(Backward_Methods):
  def get_density(strikes, prices, time_to_expiry, R, D,):
    return super().get_density(prices)