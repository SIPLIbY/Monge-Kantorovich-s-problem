import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from two_level_models.two_level_dynamic import solution
import os
from utility import calc_variance, calc_mean, test_heteroscedasticity, calculate_integral, calculate_loss


path_to_folder = "simulations"


#simulation parameters
EPOCHS = 1000


#model parameters
MARGINALITY = 0.07
PUNISHMENT_SHOP = 0.02
PUNISHMENT_STOCK = 0.01

#parameter of poisson distribution
LAMBDA = 20


values_confidence = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


values = [100]





df = pd.DataFrame()

counter = 0
total_iterations = (len(values_confidence) ** 2) * (len(values))**4

products = 100
factories = 100
stock = 100
shop = 100

for CONFIDENCE_LEVEL_SHOP in values_confidence:
    for CONFIDENCE_LEVEL_STOCK in values_confidence:
        print(f"confidence_level_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}")
        remains_shop_on_each_iteration, remain_stock_on_each_iteration, objective_values, lost_products_shop, lost_products_stock = solution(EPOCHS, products, factories, stock, shop, CONFIDENCE_LEVEL_SHOP, CONFIDENCE_LEVEL_STOCK, LAMBDA)
        
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
            loss_3 = PUNISHMENT_SHOP * sum(remains_shop_on_each_iteration[epoch, i, l] for i, l in itertools.product(range(products), range(shop)))
            loss_4 = PUNISHMENT_STOCK * sum(remain_stock_on_each_iteration[epoch, i, l] for i, l in itertools.product(range(products), range(stock)))

            losses.append(loss_1 + loss_2 + loss_3 + loss_4)

        #analyze losses
        variance_loss = calc_variance(losses)
        mean_loss = calc_mean(losses)
        integral_loss = calculate_integral(losses, 0, EPOCHS)


        #коэффициент вариации
        variation_coefficient_objective = np.sqrt(variance) / mean
        variation_coefficient_loss = np.sqrt(variance_loss) / mean_loss

        #save results of test results to csv
        #add results to dataframe
        #genereate one csv file appending to df
        temp = pd.DataFrame({
            "confidence_level_shop": [CONFIDENCE_LEVEL_SHOP],
            "confidence_level_stock": [CONFIDENCE_LEVEL_STOCK],
            "variance_objective": [variance],
            "mean_objective": [mean],
            "variation_coefficient_objective": [variation_coefficient_objective],
            "integral_objective": [integral],
            "sell_loss_objective": [loss],
            "heteroscedasticity_p_value": [heteroscedasticity_p_value],
            "heteroscedasticity_f_value": [heteroscedasticity_f_value],
            "variance_loss": [variance_loss],
            "mean_loss": [mean_loss],
            "variation_coefficient_loss": [variation_coefficient_loss],
            "integral_loss": [integral_loss]
        })
        df = pd.concat([df, temp], ignore_index=True)
        
        #plot objective values
        plt.plot(objective_values)
        plt.title(f"confidence_level_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}")
        plt.savefig(f"{path}/objective_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}.png")
        plt.close()

        #plot losses
        plt.plot(losses)
        plt.title(f"confidence_level_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}")
        plt.savefig(f"{path}/loss_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}.png")
        plt.close()
        

        counter += 1
        print(f'THATS ONLY {(counter/total_iterations) * 100}%')

df.to_csv(f"{path}/test_results.csv", index=False)






