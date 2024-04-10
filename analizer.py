import numpy as np
import itertools
import random
from ortools.linear_solver import pywraplp
import pandas as pd
import matplotlib.pyplot as plt
from two_level_dynamic_final import solution
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
from scipy import integrate
import os

path_to_folder = "simulations"


def test_heteroscedasticity(array):
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
    # Создаем массив x значений
    x = np.linspace(start, end, len(array))
    
    # Вычисляем интеграл
    integral = integrate.trapz(array, x)
    
    return integral    
    

def calculate_loss(array, marginality):
    return sum(array) * marginality




EPOCHS = 1000
AMOUNT_OF_PRODUCTS = 1
AMOUNT_OF_FACTORIES = 1
AMOUNT_OF_STOCKS = 1
AMOUNT_OF_SHOPS = 1

CONFIDENCE_LEVEL_SHOP = 0.95
CONFIDENCE_LEVEL_STOCK = 0.5

MARGINALITY = 0.07
PUNISHMENT_SHOP = 0.001
PUNISHMENT_STOCK = 0.002

lambda_param = 20
# Возможные значения параметров
values = [1, 10, 100]

# remains_shop_on_each_iteration, remain_stock_on_each_iteration, objective_values, lost_products_shop, lost_products_stock = solution(EPOCHS, AMOUNT_OF_PRODUCTS, AMOUNT_OF_FACTORIES, AMOUNT_OF_STOCKS, AMOUNT_OF_SHOPS, CONFIDENCE_LEVEL_SHOP, CONFIDENCE_LEVEL_STOCK, lambda_param)
# #plot objective values
# plt.plot(objective_values)
# plt.show()

# #variance of objective values
# variance = calc_variance(objective_values)

counter = 0

# Перебираем все возможные комбинации параметров
for products in values:
    for factories in values:
        for stock in values:
            for shop in values:
                remains_shop_on_each_iteration, remain_stock_on_each_iteration, objective_values, lost_products_shop, lost_products_stock = solution(EPOCHS, products, factories, stock, shop, CONFIDENCE_LEVEL_SHOP, CONFIDENCE_LEVEL_STOCK, lambda_param)
                
                #create folder for saving results
                path = f"{path_to_folder}/products_{products}/factories_{factories}/stock_{stock}/shop_{shop}"
                if not os.path.exists(path):
                    os.makedirs(path)
                
                variance = calc_variance(objective_values)
                mean = calc_mean(objective_values)
                integral = calculate_integral(objective_values, 0, EPOCHS)
                loss = calculate_loss(lost_products_shop, MARGINALITY)
                heteroscedasticity_p_value, heteroscedasticity_f_value = test_heteroscedasticity(objective_values)

                losses = []
                for epoch in range(EPOCHS - 1):
                    loss_1 = MARGINALITY * lost_products_shop[epoch]
                    loss_2 = MARGINALITY * lost_products_stock[epoch]
                    loss_3 = PUNISHMENT_SHOP * sum(remains_shop_on_each_iteration[epoch, i, l] for i, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_SHOPS)))
                    loss_4 = PUNISHMENT_STOCK * sum(remain_stock_on_each_iteration[epoch, i, l] for i, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS)))

                    losses.append(loss_1 + loss_2 + loss_3 + loss_4)

                #analyze losses
                variance_loss = calc_variance(losses)
                mean_loss = calc_mean(losses)
                integral_loss = calculate_integral(losses, 0, EPOCHS)

                #save results of test results to csv
                df = pd.DataFrame({
                    "variance_objective": [variance],
                    "mean_objective": [mean],
                    "integral_objective": [integral],
                    "sell_loss_objective": [loss],
                    "heteroscedasticity_p_value": [heteroscedasticity_p_value],
                    "heteroscedasticity_f_value": [heteroscedasticity_f_value],
                    "variance_loss": [variance_loss],
                    "mean_loss": [mean_loss],
                    "integral_loss": [integral_loss]
                })
                #create and save graphics with title
                plt.plot(objective_values)
                plt.title(f"OBJECTIVE_products_{products}_factories_{factories}_stock_{stock}_shop_{shop}")
                plt.savefig(f"{path}/objective_values.png")
                plt.close()

                #create and save graphics with title for losses
                plt.plot(losses)
                plt.title(f"LOSS_products_{products}_factories_{factories}_stock_{stock}_shop_{shop}")
                plt.savefig(f"{path}/loss_values.png")

                plt.close()
                df.to_csv(f"{path}/test_results.csv", index=False)

                #print progress
                print(f"products_{products}_factories_{factories}_stock_{stock}_shop_{shop} is done")
                counter += 1
                print(f'THATS ONLY {(counter/81) * 100}%')
                
                
                

