import numpy as np
import itertools
import random
from ortools.linear_solver import pywraplp
import pandas as pd

# Set the random seed
np.random.seed(42)
random.seed(42)


AMOUNT_OF_PRODUCTS = 10
AMOUNT_OF_FACTORIES = 1
AMOUNT_OF_STOCKS = 1
AMOUNT_OF_SHOPS = 1


# production cost of the product i on the factory j
def generate_production_costs():
    return np.random.randint(10, 20, (AMOUNT_OF_PRODUCTS, AMOUNT_OF_FACTORIES))

# transportation cost of the product i from the factory j to the stock k
def generate_transportation_costs_from_factory_to_stock():
    return np.random.randint(1, 10, (AMOUNT_OF_FACTORIES, AMOUNT_OF_STOCKS))

# transportation cost of the unit of product from the stock k to the shop l
def generate_transportation_costs_from_stock_to_shop():
    return np.random.randint(10, 15, (AMOUNT_OF_STOCKS, AMOUNT_OF_SHOPS))

# demand for the product i in the shop l
def generate_minimum_quantity():
    return np.random.randint(0, 4, (AMOUNT_OF_PRODUCTS, AMOUNT_OF_SHOPS))

# capacity of the stock k
def generate_capacity_stock():
    return np.random.randint(100, 200, (AMOUNT_OF_STOCKS))

# capacity of the shop l
def generate_capacity_shops():
    return np.random.randint(30, 40, (AMOUNT_OF_SHOPS))


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
capacity_stock = generate_capacity_stock()
for k in range(AMOUNT_OF_STOCKS):
    solver.Add(sum(y[i, j, k] for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES)) ) <= capacity_stock[k])

    
#capacity shop constraint
capacity_shop = generate_capacity_shops()
for l in range(AMOUNT_OF_SHOPS):
    solver.Add(sum(z[i, k, l] for i, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS)) ) <= capacity_shop[l])


#flow constraint
for i, j in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_FACTORIES)):
    solver.Add(sum(y[i, j, k] - x[i, j] for k in (range(AMOUNT_OF_STOCKS))) == 0)

#flow constraint
for i, k in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_STOCKS)):
    solver.Add(sum(z[i, k, l] for l in range(AMOUNT_OF_SHOPS)) - sum(y[i, j, k] for j in range(AMOUNT_OF_FACTORIES)) == 0)

#add demand constraint
minimum_quantity = generate_minimum_quantity()
for i, l in itertools.product(range(AMOUNT_OF_PRODUCTS), range(AMOUNT_OF_SHOPS)):
    solver.Add(sum(z[i, k, l] for k in range(AMOUNT_OF_STOCKS)) >= minimum_quantity[i, l])


#objective function
production_costs = generate_production_costs()
transportation_costs_from_factory_to_stock = generate_transportation_costs_from_factory_to_stock()
transportation_costs_from_stock_to_shop = generate_transportation_costs_from_stock_to_shop()


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

    
print("All input data")
print("Production costs")
print(production_costs)
print("transportation costs from factory to stock")
print(transportation_costs_from_factory_to_stock)
print("transportation costs from stock to shop")
print(transportation_costs_from_stock_to_shop)
print("minimum quantity")
print(minimum_quantity)
print("capacity stock")
print(capacity_stock)
print("capacity shops")
print(capacity_shop)




