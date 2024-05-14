import random
from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
import itertools
random.seed(42)
np.random.seed(42)



#DATA GENERATION
columns_shipping = ['DC_id', 'supply_node_id', 'cost']
columns_supply_node = ['supply_node_id', 'capacity']
column_itemsets = ['itemset_id', 'sku_id']
columns_supply = ['node_id', 'sku_id', 'cost', 'current_quantity']
columns_demand_nodes = ['node_id', 'itemset_id', 'demand_mean', 'demand_variance']


def generate_shipping(amount_DC, amount_supply_node, min_cost=5, max_cost=30):
    DC_list = np.arange(amount_DC)
    supply_node_list = np.arange(amount_supply_node)
    cost_matrix = np.random.randint(min_cost, max_cost, size=(amount_DC * amount_supply_node, 1))

    all_ways_matrix = np.array(list(itertools.product(DC_list, supply_node_list)))
    result = np.hstack((all_ways_matrix, cost_matrix))
    return pd.DataFrame(result, columns=columns_shipping)



def generate_supply_nodes(amount_supply_node, min_capacity=10, max_capacity=20):
    supply_node_list = np.arange(amount_supply_node)
    capacity_array = np.random.randint(min_capacity, max_capacity, size=(amount_supply_node,))
    result = np.vstack((supply_node_list, capacity_array)).T
    return pd.DataFrame(result, columns=columns_supply_node)
#print(generate_supply_nodes(3), '\n')
def generate_itemsets(amount_itemsets, amount_sku, p = 0.2):
    itemset_list = np.arange(amount_itemsets)
    sku_list = np.arange(amount_sku)
    all_ways_mat = list(itertools.product(itemset_list, sku_list))
    all_ways_mat = np.array(random.sample(all_ways_mat, int(len(all_ways_mat) * p)))
    return pd.DataFrame(all_ways_mat, columns =column_itemsets).sort_values("itemset_id").reset_index(drop=True)
#print(generate_itemsets(1, 2))

def generate_supply(supply_nodes, itemsets, min_procurement_cost, max_procurement_cost):

    skus = pd.DataFrame(itemsets.sku_id.unique(), columns = ['sku_id'])
    skus['cost'] = np.random.randint(min_procurement_cost, max_procurement_cost, size=skus.shape)
    merged = supply_nodes.merge(skus,  how='cross')
    merged['cost'] = merged['cost'] + np.random.randint(0, 8, size=merged['cost'].shape)
    merged['current_quantity'] = 0
    sku_amount = itemsets.sku_id.unique().shape[0]
    supply_node_list = merged.supply_node_id.unique()

    for i in supply_node_list:
        capacity_temp = merged[merged.supply_node_id == i]['capacity'].iloc[0]
        current_storage = np.random.randint(capacity_temp)
        get_each_sku_cap = np.random.multinomial(current_storage, [1 / sku_amount] * sku_amount)
        merged.loc[merged.supply_node_id == i, 'current_quantity'] = get_each_sku_cap
    #print(merged)
    return merged.drop(['capacity'], axis=1)

#print(generate_supply(generate_supply_nodes(3), generate_itemsets(2, 3, p=1), 10, 100))


def generate_demand_nodes(demand_node_amount, itemsets, min_mean_demand=5, max_mean_demand=15, min_var_demand=1, max_var_demand=4):
    demand_node_list = np.arange(demand_node_amount)
    itemset_list = itemsets.itemset_id.unique()

    result = np.array(list(itertools.product(demand_node_list, itemset_list)))
    demand_mean_array = np.random.randint(min_mean_demand, max_mean_demand, size=(result.shape[0], 1))
    demand_var_array = np.round(np.random.uniform(min_var_demand, max_var_demand, size=(result.shape[0], 1)), 2)
    result = np.hstack((result, demand_mean_array, demand_var_array))
    result = pd.DataFrame(result, columns=columns_demand_nodes)
    result[['node_id', 'itemset_id', 'demand_mean']] = result[['node_id', 'itemset_id', 'demand_mean']].astype(int)
    return pd.DataFrame(result, columns=columns_demand_nodes)



SUPPLY_NODE_NUMBER = 1
DEMAND_NODE_NUMBER = 1

MIN_SHIPPING_COST = 5
MAX_SHIPPING_COST = 10


MIN_CAPACITY_SUPPLY_NODE = 4
MAX_CAPACITY_SUPPLY_NODE = 10

ITEMSET_NUMBER = 1
SKU_NUMBER = 2

MIN_PROCUREMENT_COST = 10
MAX_PROCUREMENT_COST = 50

MIN_MEAN_DEMAND = 2
MAX_MEAN_DEMAND = 5

MIN_VAR_DEMAND = 1
MAX_VAR_DEMAND = 2

# shipping = generate_shipping(SUPPLY_NODE_NUMBER, DEMAND_NODE_NUMBER, MIN_SHIPPING_COST, MAX_SHIPPING_COST)
# itemsets = generate_itemsets(ITEMSET_NUMBER, SKU_NUMBER, p=1 )
# supply_nodes = generate_supply_nodes(SUPPLY_NODE_NUMBER, MIN_CAPACITY_SUPPLY_NODE, MAX_CAPACITY_SUPPLY_NODE)
# supply = generate_supply(supply_nodes, itemsets, MIN_PROCUREMENT_COST, MAX_PROCUREMENT_COST)
# demand_nodes = generate_demand_nodes(DEMAND_NODE_NUMBER, itemsets, MIN_MEAN_DEMAND, MAX_MEAN_DEMAND, MIN_VAR_DEMAND, MAX_VAR_DEMAND)

shipping = pd.DataFrame({"DC_id":[0],  "supply_node_id":[0],  "cost":[3]})
itemsets = pd.DataFrame({"itemset_id":[0],  "sku_id":[0]})
supply_nodes = pd.DataFrame({"supply_node_id":[0],"capacity":[30]})
supply = pd.DataFrame({"node_id":[0],  "sku_id":[0],  "cost":[3], "current_quantity":[3]})
demand_nodes = pd.DataFrame({"node_id":[0], "itemset_id":[0], "demand_mean":[10], "demand_variance":[0]})

#print all values
# print("SHIPPING", "\n",  shipping)
# print("ITEMSETS","\n", itemsets)
# print('SUPPLY_NODES', "\n",supply_nodes)
# print("SUPPLY", "\n",supply)
# print("DEMAND_NODES","\n", demand_nodes)



SUPPLY_NODE_LIST = list(supply_nodes.supply_node_id.unique())
DEMAND_NODE_LIST = list(demand_nodes.node_id.unique())
ITEMSET_LIST = list(itemsets.itemset_id.unique())
SKU_LIST = list(itemsets.sku_id.unique())

SUPPLY_NODE_NUMBER_CONST = len(SUPPLY_NODE_LIST)
DEMAND_NODE_NUMBER_CONST = len(DEMAND_NODE_LIST)
ITEMSET_NUMBER_CONST = len(ITEMSET_LIST)
SKU_NUMBER_CONST = len(SKU_LIST)

def get_solver():
    solver =pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        exit("No Solver")
    return solver

solver = get_solver()


x = {}
y = {}
z = {}
v = {}

infty = solver.infinity()
for i, j, k in itertools.product(SUPPLY_NODE_LIST, DEMAND_NODE_LIST, ITEMSET_LIST):
    x[i, j, k] = solver.IntVar(0, infty, f'x[{i}][{j}][{k}]')
    z[i, j, k] = solver.NumVar(0, 1, f'z[{i}][{j}][{k}]')

for i, r in itertools.product(SUPPLY_NODE_LIST, SKU_LIST):
    y[i, r] = solver.IntVar(0, infty, f'y[{i}][{r}]')
    


for j, k in itertools.product(DEMAND_NODE_LIST, ITEMSET_LIST):
    constraint = solver.RowConstraint(1, 1, f'equation 1: (j, k) = ({j}, {k})')
    for i in SUPPLY_NODE_LIST:
      constraint.SetCoefficient(z[i, j, k], 1)


demand_node_dict = demand_nodes.set_index(['node_id', 'itemset_id']).to_dict('index')

for i, j, k in itertools.product(SUPPLY_NODE_LIST, DEMAND_NODE_LIST, ITEMSET_LIST):
    constraint = solver.RowConstraint(-infty, 0, f'equation 2: z({i},{j},{k})')
    demand_j_k = demand_node_dict[j, k]
    demand_int = demand_j_k['demand_mean']
    constraint.SetCoefficient(z[i, j, k], demand_int)
    constraint.SetCoefficient(x[i, j, k], -1)

itemset_sku_tuples = list(itemsets.itertuples(index=False, name=None))
supply_dict = supply.set_index(['node_id', 'sku_id']).to_dict('index')

for i, r in itertools.product(SUPPLY_NODE_LIST, SKU_LIST):
    constraint = solver.RowConstraint(-infty, supply_dict[i, r]['current_quantity'], 'equation 3:')

    for j, k in itertools.product(DEMAND_NODE_LIST, ITEMSET_LIST):
        if (k, r) in itemset_sku_tuples:
            constraint.SetCoefficient(x[i, j, k], 1)
    constraint.SetCoefficient(y[i, r], -1)


for i, r in itertools.product(SUPPLY_NODE_LIST, SKU_LIST):
    constraint = solver.RowConstraint(-infty, -supply_dict[i, r]['current_quantity'], 'equation 4:')
    for j, k in itertools.product(DEMAND_NODE_LIST, ITEMSET_LIST):
        if (k, r) in itemset_sku_tuples:
            constraint.SetCoefficient(x[i, j, k], -1)

    

item_set_sku_id_count = itemsets.groupby(['itemset_id'])['itemset_id'].count().to_dict()
supply_node_capacity = supply_nodes.set_index('supply_node_id').to_dict('index')

for i in SUPPLY_NODE_LIST:
    constraint = solver.RowConstraint(-infty, supply_node_capacity[i]['capacity'])
    for j, k in itertools.product(DEMAND_NODE_LIST, ITEMSET_LIST):
        constraint.SetCoefficient(x[i, j, k], item_set_sku_id_count[k])


for i, r in itertools.product(SUPPLY_NODE_LIST, SKU_LIST):
    constraint = solver.RowConstraint(supply_dict[i, r]['current_quantity'], supply_dict[i, r]['current_quantity'], 'equation 4:')
    for j, k in itertools.product(DEMAND_NODE_LIST, ITEMSET_LIST):
        if (k, r) in itemset_sku_tuples:
            constraint.SetCoefficient(x[i, j, k], 1)
    constraint.SetCoefficient(y[i, r], -1)
    

objective = solver.Objective()

item_set_sku_id_count = itemsets.groupby(['itemset_id'])['itemset_id'].count().to_dict()
shipping_cost_dict = shipping.set_index(['DC_id', 'supply_node_id']).to_dict('index')
supply_dict = supply.set_index(['node_id', 'sku_id']).to_dict('index')

for i, j, k in itertools.product(SUPPLY_NODE_LIST, DEMAND_NODE_LIST, ITEMSET_LIST):
    objective.SetCoefficient(x[i, j, k], item_set_sku_id_count[k]*shipping_cost_dict[i, j]['cost'])

for i, r in itertools.product(SUPPLY_NODE_LIST, SKU_LIST):
    objective.SetCoefficient(y[i, r], supply_dict[i, r]['cost'])

objective.SetMinimization()
print('Number of constraints =', solver.NumConstraints())




#PRINTING RESULTS
status = solver.Solve()

#### SET HERE
print_only_nonzero = False

if status == pywraplp.Solver.OPTIMAL:
    print("There is an optimal solution")

    print(f"Objective value: {objective.Value()}")

    eps = 1e-7
    if solver.VerifySolution(eps, True):
        print(f"Solution is verified and correct within the tolerance of {eps}")
    else:
        print(f"Solution is not verified, absolute error is bigger than {eps}")

    for i, j, k in itertools.product(SUPPLY_NODE_LIST, DEMAND_NODE_LIST, ITEMSET_LIST):

        if (not print_only_nonzero) | (x[i, j, k].solution_value() != 0):
            print(x[i, j, k].name(), " = ", x[i, j, k].solution_value())
    print()

    for i, j, k in itertools.product(SUPPLY_NODE_LIST, DEMAND_NODE_LIST, ITEMSET_LIST):

        if (not print_only_nonzero) | (z[i, j, k].solution_value() != 0):
            print(z[i, j, k].name(), " = ", z[i, j, k].solution_value())
    print()

    for i, r in itertools.product(SUPPLY_NODE_LIST, SKU_LIST):

        if (not print_only_nonzero) | (y[i, r].solution_value() != 0):
            print(y[i, r].name(), " = ", y[i, r].solution_value())
    print()


else:
    print("No optimal solution")



