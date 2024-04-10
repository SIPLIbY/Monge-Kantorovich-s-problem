#https://ru.overleaf.com/project/64233d28ca81fd4da7f1afad

import numpy as np
import itertools
import random
from ortools.linear_solver import pywraplp
import pandas as pd
from utility import generate_production_costs, generate_transportation_costs_from_factory_to_stock, generate_transportation_costs_from_stock_to_shop, generate_minimum_quantity, generate_capacity_stock, generate_capacity_shops
from utility import get_estimation
import matplotlib.pyplot as plt



def solution(epochs, amount_of_products, amount_of_factories, amount_of_stocks, amount_of_shops, confidence_level_shop, confidence_level_stock, _lambda_param):
    # Set the random seed
    np.random.seed(43)
    random.seed(43)

    EPOCHS = epochs
    AMOUNT_OF_PRODUCTS = amount_of_products
    AMOUNT_OF_FACTORIES = amount_of_factories
    AMOUNT_OF_STOCKS = amount_of_stocks
    AMOUNT_OF_SHOPS = amount_of_shops

    CONFIDENCE_LEVEL_SHOP = confidence_level_shop
    CONFIDENCE_LEVEL_STOCK = confidence_level_stock

    lambda_param = _lambda_param


    # production cost of the product i on the factory j
    production_costs = generate_production_costs(AMOUNT_OF_PRODUCTS, AMOUNT_OF_FACTORIES)

    # transportation cost of the product i from the factory j to the stock k
    transportation_costs_from_factory_to_stock = generate_transportation_costs_from_factory_to_stock(AMOUNT_OF_FACTORIES, AMOUNT_OF_STOCKS)

    # transportation cost of the unit of product from the stock k to the shop l
    transportation_costs_from_stock_to_shop = generate_transportation_costs_from_stock_to_shop(AMOUNT_OF_STOCKS, AMOUNT_OF_SHOPS)


    # capacity of the stock k
    capacity_stock = generate_capacity_stock(AMOUNT_OF_STOCKS)


    #at the beginning we have zero remains_shop
    remains_shop = np.zeros((AMOUNT_OF_PRODUCTS, AMOUNT_OF_SHOPS))
    remain_stock = np.zeros((AMOUNT_OF_PRODUCTS, AMOUNT_OF_STOCKS))

    remains_shop_on_each_iteration = []
    remain_stock_on_each_iteration = []

    unsold_products_shop = []
    unsold_products_stock = []

    objective_values = []

    demand_history_for_shop = []
    demand_history_for_stock = []

    demand_estimation_stock = 0

    previous_supply = {}


    for epoch in range(EPOCHS):
        #get real demand
        demand = generate_minimum_quantity(AMOUNT_OF_PRODUCTS, AMOUNT_OF_SHOPS, lambda_param)
        demand_history_for_shop.append(np.asarray(demand).mean())
        mean_value = round(np.asarray(demand_history_for_shop.copy()).mean())
        demand_estimation_shop = get_estimation(mean_value, CONFIDENCE_LEVEL_SHOP)


        demand_estimation_stock = round(np.percentile(demand_history_for_shop, CONFIDENCE_LEVEL_STOCK*100) if len(demand_history_for_shop) != 0 else demand_estimation_shop)
        

        #check and update the remains_shop
        #if epoch is first so dont update the remains_shop
        if epoch != 0:
            
            for i, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_SHOPS)):
                remains_shop[i, l] += sum(previous_supply[i, k, l] for k in range(AMOUNT_OF_STOCKS)) - demand[i, l]
                
                if remains_shop[i, l] < 0:
                    unsold_products_shop.append(abs(remains_shop[i, l]))
                else:
                    unsold_products_shop.append(0)

                remains_shop[i, l] = 0 if remains_shop[i, l] < 0 else remains_shop[i, l]

        def get_solver():
            solver =pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                exit("No Solver")
            return solver

        solver = get_solver()

        infty = solver.infinity()

        x = {}
        y = {}
        z = {}

        #defining the variables
        for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES)):
            x[i, j] = solver.IntVar(0, infty, f"x_{i}_{j}")

        for i, j, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES), range(AMOUNT_OF_STOCKS)):
            y[i, j, k] = solver.IntVar(0, infty, f"y_{i}_{j}_{k}")

        for i, k, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS), range(AMOUNT_OF_SHOPS)):
            z[i, k, l] = solver.IntVar(0, infty, f"z_{i}_{k}_{l}")


        #capacity stock constraint 
        for k in range(AMOUNT_OF_STOCKS):
            solver.Add(sum(y[i, j, k] for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES))) + sum(remain_stock[i, k] for i in range(AMOUNT_OF_PRODUCTS)) <= capacity_stock[k])

        #demand constraint for the stock
        for i in range(AMOUNT_OF_PRODUCTS):
            solver.Add(sum(y[i, j, k] for j, k in itertools.product(range(AMOUNT_OF_FACTORIES), range(AMOUNT_OF_STOCKS))) + sum(remain_stock[i, k] for k in range(AMOUNT_OF_STOCKS)) >= demand_estimation_stock)
        #flow constraint
        for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES)):
            solver.Add(sum(y[i, j, k] for k in (range(AMOUNT_OF_STOCKS))) - x[i, j]  == 0)

        #flow constraint
        for i, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS)):
            solver.Add(sum(y[i, j, k] for j in range(AMOUNT_OF_FACTORIES)) + remain_stock[i, k] - sum(z[i, k, l] for l in range(AMOUNT_OF_SHOPS)) >=0)


        #add demand constraint
        for i, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_SHOPS)):
            solver.Add(sum(z[i, k, l] for k in range(AMOUNT_OF_STOCKS)) + remains_shop[i,l] >= demand_estimation_shop)

        
        
        #define objective function
        objective = sum(production_costs[i][j] * x[i, j] for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES))) + \
                    sum(transportation_costs_from_factory_to_stock[j, k] * y[i, j, k] for i, j, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES), range(AMOUNT_OF_STOCKS))) + \
                    sum(transportation_costs_from_stock_to_shop[k, l] * z[i, k, l] for i, k, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS), range(AMOUNT_OF_SHOPS)))

        #minimize objective function
        solver.Minimize(objective)
        status = solver.Solve()

        # print(f"EPOCH {epoch+1}")

        # for k in range(AMOUNT_OF_STOCKS):
        #     for i in range(AMOUNT_OF_PRODUCTS):
        #         print(f"Remains of product {i} on the stock {k} ==", remain_stock[i, k], 'BEFORE')


        # for i in range(AMOUNT_OF_PRODUCTS):
        #     print(f"Ordered total {sum(y[i, j, k].solution_value() for j, k in itertools.product(range(AMOUNT_OF_FACTORIES), range(AMOUNT_OF_STOCKS)))} product {i} from factories")
        #     print(f"Shops ordered total {sum(z[i, k, l].solution_value() for k, l in itertools.product(range(AMOUNT_OF_STOCKS), range(AMOUNT_OF_SHOPS)))} product {i} from stocks")

        #update remains_stock
        for i, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS)):
            remain_stock[i, k] += sum(y[i, j, k].solution_value() for j in range(AMOUNT_OF_FACTORIES)) - sum(z[i, k, l].solution_value() for l in range(AMOUNT_OF_SHOPS))
            if remain_stock[i, k] < 0:
                unsold_products_stock.append(abs(remain_stock[i, k]))
                remain_stock[i, k] = 0
            else:
                unsold_products_stock.append(0)
        
        # for i,k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS)):
        #     print(f"Remains_stock product {i} in stock {k} is {remain_stock[i, k]}")

        supply_i = sum(z[i, k, l].solution_value() for k, l in itertools.product(range(AMOUNT_OF_STOCKS), range(AMOUNT_OF_SHOPS)))
        demand_history_for_stock.append(np.asarray(supply_i).mean())
        
        # print("Demand estimation stock", demand_estimation_stock)
        # for i, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_SHOPS)):
        #     print(f"Remains_shop product {i} in shop {l} is {remains_shop[i, l]}")
        #     print(f"Actual demand of product {i} in the shop{l} is ", demand[i, l])
        #     print(f"Total delivered product {i} to the shop {l} is ", sum(previous_supply[i, k, l] for k in range(AMOUNT_OF_STOCKS)) if epoch > 0 else "N/A")
        #     print(f"Current solution to deliver product {i} to the shop {l} with amount", sum(z[i, k, l].solution_value() for k in range(AMOUNT_OF_STOCKS)))
        #     print(f"Current mean demand of product {i} is ", mean_value)
        #     print(f"Current estimation of demand of product {i} is ", demand_estimation_shop)
        #     print('\n')
        
        # print("\n\n")

        
        #add remains_shop to the list
        remains_shop_on_each_iteration.append(remains_shop.copy())
        remain_stock_on_each_iteration.append(remain_stock.copy())
        

        
        
        objective_values.append(solver.Objective().Value())
        #to add last remains_shop
        if epoch == EPOCHS - 1:
            remains_shop_on_each_iteration.append(remains_shop.copy())

        for i,k,l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS), range(AMOUNT_OF_SHOPS)):
            previous_supply[i, k, l] = z[i, k, l].solution_value()



    remains_shop_on_each_iteration = np.array(remains_shop_on_each_iteration)
    remain_stock_on_each_iteration = np.array(remain_stock_on_each_iteration)
    objective_values = np.array(objective_values)

    return remains_shop_on_each_iteration, remain_stock_on_each_iteration, objective_values, unsold_products_shop, unsold_products_stock









