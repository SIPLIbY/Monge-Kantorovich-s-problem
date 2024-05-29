import scipy.stats as stats
from scipy.stats import poisson
import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
from scipy import integrate


def get_estimation(lambda_param, alpha):
    return stats.poisson.isf(1 - alpha, lambda_param)


def generate_production_costs(Amount_of_products, Amount_of_factories):
    return np.random.randint(1500, 2000, (Amount_of_products, Amount_of_factories))

def generate_transportation_costs_from_factory_to_stock(Amount_of_factories, Amount_of_stocks):
    return np.random.randint(2000, 3000, (Amount_of_factories, Amount_of_stocks))

def generate_transportation_costs_from_stock_to_shop(Amount_of_stocks, Amount_of_shops):
    return np.random.randint(300, 1000, (Amount_of_stocks, Amount_of_shops))

def generate_minimum_quantity(Amount_of_products, Amount_of_shops, lambda_param=10):
    return np.random.poisson(lambda_param, (Amount_of_products, Amount_of_shops))

def generate_capacity_stock(Amount_of_stocks):
    return np.random.randint(30000, 50000, (Amount_of_stocks))

def generate_capacity_shops(Amount_of_shops):
    return np.random.randint(300000, 400000, (Amount_of_shops))

def generate_remains(Amount_of_products, Amount_of_shops):
    return np.random.randint(0, 5, (Amount_of_products, Amount_of_shops))

def generate_factory_limit(Amount_of_factories):
    return np.random.randint(500, 1000, (Amount_of_factories))



def test_heteroscedasticity(array):
    #delete first 10 values
    array = array[10:]
    y = array
    x = np.arange(0, len(y), 1)
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    
    res = het_breuschpagan(results.resid, results.model.exog)
    
    
    
    return res[1], res[3]

def calc_variance(array):
    return np.var(array)


def calc_mean(array):
    return np.mean(array)


def calculate_integral(array, start, end):
    x = np.linspace(start, end, len(array))
    integral = integrate.trapz(array, x)    
    
    return integral    
    

def calculate_loss(array, marginality):
    return sum(array) * marginality