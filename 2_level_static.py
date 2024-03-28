#https://ru.overleaf.com/project/64233d28ca81fd4da7f1afad

import numpy as np
import itertools
import random
from ortools.linear_solver import pywraplp
import pandas as pd
from utility import generate_production_costs, generate_transportation_costs_from_factory_to_stock, generate_transportation_costs_from_stock_to_shop, generate_minimum_quantity, generate_capacity_stock, generate_capacity_shops
import matplotlib.pyplot as plt
# Set the random seed
# np.random.seed(43)
# random.seed(43)

EPOCHS = 100

AMOUNT_OF_PRODUCTS = 2
AMOUNT_OF_FACTORIES = 10
AMOUNT_OF_STOCKS = 10
AMOUNT_OF_SHOPS = 1
lambda_param = 15

objective_values = []
for i in range (EPOCHS):
    # production cost of the product i on the factory j
    production_costs = generate_production_costs(AMOUNT_OF_PRODUCTS, AMOUNT_OF_FACTORIES)

    # transportation cost of the product i from the factory j to the stock k
    transportation_costs_from_factory_to_stock = generate_transportation_costs_from_factory_to_stock(AMOUNT_OF_FACTORIES, AMOUNT_OF_STOCKS)

    # transportation cost of the unit of product from the stock k to the shop l
    transportation_costs_from_stock_to_shop = generate_transportation_costs_from_stock_to_shop(AMOUNT_OF_STOCKS, AMOUNT_OF_SHOPS)

    # demand for the product i in the shop l
    minimum_quantity = generate_minimum_quantity(AMOUNT_OF_PRODUCTS, AMOUNT_OF_SHOPS, 15)

    # capacity of the stock k
    capacity_stock = generate_capacity_stock(AMOUNT_OF_STOCKS)

    # capacity of the shop l
    capacity_shop = generate_capacity_shops(AMOUNT_OF_SHOPS)



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
        solver.Add(sum(y[i, j, k] for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES)) ) <= capacity_stock[k])

        
    #capacity shop constraint
    for l in range(AMOUNT_OF_SHOPS):
        solver.Add(sum(z[i, k, l] for i, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS)) ) <= capacity_shop[l])


    #flow constraint
    for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES)):
        solver.Add(sum(y[i, j, k] for k in (range(AMOUNT_OF_STOCKS))) - x[i, j]  == 0)

    #flow constraint
    for i, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS)):
        solver.Add(sum(z[i, k, l] for l in range(AMOUNT_OF_SHOPS)) - sum(y[i, j, k] for j in range(AMOUNT_OF_FACTORIES)) == 0)

    #add demand constraint
    for i, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_SHOPS)):
        solver.Add(sum(z[i, k, l] for k in range(AMOUNT_OF_STOCKS)) >= minimum_quantity[i, l])


    #objective function


    #define objective function
    objective = sum(production_costs[i][j] * x[i, j] for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES))) + \
                sum(transportation_costs_from_factory_to_stock[j, k] * y[i, j, k] for i, j, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES), range(AMOUNT_OF_STOCKS))) + \
                sum(transportation_costs_from_stock_to_shop[k, l] * z[i, k, l] for i, k, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS), range(AMOUNT_OF_SHOPS)))

    #minimize objective function
    solver.Minimize(objective)

    #print optimal value
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print('Objective value =', solver.Objective().Value())
        # print the optimal solution
        for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES)):
            print(f"x_{i}_{j} = {x[i, j].solution_value()}")

        for i, j, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES), range(AMOUNT_OF_STOCKS)):
            print(f"y_{i}_{j}_{k} = {y[i, j, k].solution_value()}")

        for i, k, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS), range(AMOUNT_OF_SHOPS)):
            print(f"z_{i}_{k}_{l} = {z[i, k, l].solution_value()}")
    else:
        print('NO SOLUTION')
        
    objective_values.append(solver.Objective().Value())


print(f"Mean of the objective values: {np.mean(objective_values)}")

# Draw graphic of the objective values
plt.plot(objective_values)
plt.xlabel('Epoch')
plt.ylabel('Objective value')
plt.title('Objective values')
plt.show()





