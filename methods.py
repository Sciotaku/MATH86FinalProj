import numpy as np 
import pandas as pd
from abc import ABC, abstractmethod


'''
Standardized inputs:
    - 
'''
class Forward_Method(ABC):
  @abstractmethod
  def get_option_prices():
    pass


class Black_Scholes(Forward_Method):
  def get_options_prices():
    pass

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
  def get_density(strikes, prices):
    pass

class Butterfly(Backward_Methods):
  def get_density(strikes, prices):
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
    return butterfly_prices
    
class Breeden_Litzenberger(Backward_Methods):
  def get_density(strikes, prices):
    pass