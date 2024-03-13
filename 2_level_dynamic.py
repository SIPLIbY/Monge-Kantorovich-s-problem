#https://ru.overleaf.com/project/64233d28ca81fd4da7f1afad

import numpy as np
import itertools
import random
from ortools.linear_solver import pywraplp
import pandas as pd
from utility import generate_production_costs, generate_transportation_costs_from_factory_to_stock, generate_transportation_costs_from_stock_to_shop, generate_minimum_quantity, generate_capacity_stock, generate_capacity_shops
from utility import generate_remains
from utility import get_estimation
import matplotlib.pyplot as plt

# Set the random seed
np.random.seed(43)
random.seed(43)

EPOCHS = 15
AMOUNT_OF_PRODUCTS = 2
AMOUNT_OF_FACTORIES = 10
AMOUNT_OF_STOCKS = 10
AMOUNT_OF_SHOPS = 5

CONFIDENCE_LEVEL = 0.99

lambda_param = 15

demand_estimation = get_estimation(lambda_param, CONFIDENCE_LEVEL)

# production cost of the product i on the factory j
production_costs = generate_production_costs(AMOUNT_OF_PRODUCTS, AMOUNT_OF_FACTORIES)

# transportation cost of the product i from the factory j to the stock k
transportation_costs_from_factory_to_stock = generate_transportation_costs_from_factory_to_stock(AMOUNT_OF_FACTORIES, AMOUNT_OF_STOCKS)

# transportation cost of the unit of product from the stock k to the shop l
transportation_costs_from_stock_to_shop = generate_transportation_costs_from_stock_to_shop(AMOUNT_OF_STOCKS, AMOUNT_OF_SHOPS)


# capacity of the stock k
capacity_stock = generate_capacity_stock(AMOUNT_OF_STOCKS)

#remains of the product i in the shop l
# remains = generate_remains(AMOUNT_OF_PRODUCTS, AMOUNT_OF_SHOPS)

#at the beginning we have zero remains
remains = np.zeros((AMOUNT_OF_PRODUCTS, AMOUNT_OF_SHOPS))

remains_on_each_iteration = []
objective_values = []

for epoch in range(EPOCHS):

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


    #flow constraint
    for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES)):
        solver.Add(sum(y[i, j, k] for k in (range(AMOUNT_OF_STOCKS))) - x[i, j]  == 0)

    #flow constraint
    for i, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS)):
        solver.Add(sum(z[i, k, l] for l in range(AMOUNT_OF_SHOPS)) - sum(y[i, j, k] for j in range(AMOUNT_OF_FACTORIES)) == 0)

    #add demand constraint
    for i, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_SHOPS)):
        solver.Add(sum(z[i, k, l] for k in range(AMOUNT_OF_STOCKS)) + remains[i,l] >= demand_estimation)




    #define objective function
    objective = sum(production_costs[i][j] * x[i, j] for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES))) + \
                sum(transportation_costs_from_factory_to_stock[j, k] * y[i, j, k] for i, j, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES), range(AMOUNT_OF_STOCKS))) + \
                sum(transportation_costs_from_stock_to_shop[k, l] * z[i, k, l] for i, k, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS), range(AMOUNT_OF_SHOPS)))

    #minimize objective function
    solver.Minimize(objective)

    status = solver.Solve()

    #get real demand
    demand = generate_minimum_quantity(AMOUNT_OF_PRODUCTS, AMOUNT_OF_SHOPS, lambda_param)

    z_copy = z.copy()

    #add remains to the list
    remains_on_each_iteration.append(remains.copy())

    print(f"EPOCH {epoch+1}")
    for i, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_SHOPS)):
        print(f"Remains product {i} in shop {l} is {remains[i, l]}")
        print(f"Actual demand of product {i} in the shop is {l}", demand[i, l])
        print(f"Total delivered product {i} to the shop {l} is ", sum(z_copy[i, k, l].solution_value() for k in range(AMOUNT_OF_STOCKS)))
        print('\n')
    
    print("\n\n")
    #check and update the remains
    for i, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_SHOPS)):
        remains[i, l] += sum(z_copy[i, k, l].solution_value() for k in range(AMOUNT_OF_STOCKS)) - demand[i, l]
        
        remains[i, l] = 0 if remains[i, l] < 0 else remains[i, l]
    
    objective_values.append(solver.Objective().Value())
    #to add last remains
    if epoch == EPOCHS - 1:
        remains_on_each_iteration.append(remains.copy())



#plot the remains
#axis x is the epoch, axis y is the product i in the shop l
remains_on_each_iteration = np.array(remains_on_each_iteration)
plt.figure()
for i in range(AMOUNT_OF_PRODUCTS):
    for l in range(AMOUNT_OF_SHOPS):
        plt.plot(remains_on_each_iteration[:, i, l], label=f"Product {i} in shop {l}")
plt.legend()


#plot the objective function
plt.figure()
plt.plot(objective_values)
plt.title("Objective function")
plt.xlabel("Epoch")
plt.ylabel("Objective value")
plt.show()





