import numpy as np 

def generate_local_volatility_surface(S, strikes, T, r, model='constant', params=None):
    """Generate local volatility for given strikes based on the specified model."""
    sigma_surface = []
    
    if model == 'constant':
        sigma = params['sigma']
        sigma_surface = [sigma] * len(strikes)
    
    elif model == 'normal':
        sigma_0 = params['sigma_0']
        S_0 = params['S_0']
        sigma_param = params['sigma_param']
        sigma_surface = [sigma_0 * np.exp(-((strike - S_0)**2 / (2 * sigma_param**2))) for strike in strikes]
    
    elif model == 'poisson':
        lambda_param = params['lambda']
        k_values = np.random.poisson(lambda_param, len(strikes))  # Poisson increments
        sigma_0 = params['sigma_0']
        sigma_surface = [sigma_0 + k for k in k_values]
    
    elif model == 'exponential':
        sigma_0 = params['sigma_0']
        beta = params['beta']
        sigma_surface = [sigma_0 * np.exp(-beta * strike) for strike in strikes]
    
    return sigma_surface