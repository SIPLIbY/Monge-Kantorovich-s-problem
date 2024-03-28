import scipy.stats as stats
from scipy.stats import poisson
import numpy as np


def get_estimation(lambda_param, alpha):
    return stats.poisson.isf(1 - alpha, lambda_param)


def generate_production_costs(Amount_of_products, Amount_of_factories):
    return np.random.randint(10, 20, (Amount_of_products, Amount_of_factories))

def generate_transportation_costs_from_factory_to_stock(Amount_of_factories, Amount_of_stocks):
    return np.random.randint(10, 20, (Amount_of_factories, Amount_of_stocks))

def generate_transportation_costs_from_stock_to_shop(Amount_of_stocks, Amount_of_shops):
    return np.random.randint(3, 9, (Amount_of_stocks, Amount_of_shops))

def generate_minimum_quantity(Amount_of_products, Amount_of_shops, lambda_param=10):
    return np.random.poisson(lambda_param, (Amount_of_products, Amount_of_shops))

def generate_capacity_stock(Amount_of_stocks):
    return np.random.randint(30, 50, (Amount_of_stocks))

def generate_capacity_shops(Amount_of_shops):
    return np.random.randint(30, 40, (Amount_of_shops))

def generate_remains(Amount_of_products, Amount_of_shops):
    return np.random.randint(0, 5, (Amount_of_products, Amount_of_shops))


