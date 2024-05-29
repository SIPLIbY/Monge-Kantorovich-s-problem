import pandas as pd
from two_level_static import solution
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


values = [1, 10]


def find_optimal_level():
    for products in values:
        for factories in values:
            for stock in values:
                for shop in values:
                    path = f"simulations/products_{products}/factories_{factories}/stock_{stock}/shop_{shop}"

                    #open csv file
                    df = pd.read_csv(f"{path}/test_results.csv")
                    #find min value in column integral_loss and print this value and print confidence_level_shop and confidence_level_stock in this row
                    min_value = df['integral_loss'].min()
                    print(f"({products},{factories},{stock},{shop})")
                    print("MIN VALUE:", min_value, "alpha:", df[df['integral_loss'] == min_value]['confidence_level_shop'].values[0], "beta:", df[df['integral_loss'] == min_value]['confidence_level_stock'].values[0])


def model_comparing():
    dif = []
    for products in values:
        for factories in values:
            for stock in values:
                for shop in values:
                    path = f"simulations/products_{products}/factories_{factories}/stock_{stock}/shop_{shop}"
                    #open csv file
                    df = pd.read_csv(f"{path}/test_results.csv")

                    print(f"({products},{factories},{stock},{shop})")
                    #find mean value in column mean_objective
                    mean_objective = df['mean_objective'].mean()
                    print("MEAN OBJECTIVE DYNAMIC:", mean_objective)

                    mean_static = solution(products, factories, stock, shop)
                    print("MEAN OBJECTIVE STATIC", mean_static)

                    #difference between mean_objective and mean_static in %
                    difference = 100 * (mean_objective - mean_static) / mean_objective
                    print("DIFFERENCE:", difference)
                    dif.append(difference)
                    print("\n")
    print("MEAN DIFFERENCE:", np.mean(dif))
    

def mean_heteroscedasticity_p_value():
    products = 10
    factories = 10
    stock = 1
    shop = 1
    # for products in values:
    #     for factories in values:
    #         for stock in values:
    #             for shop in values:
    path = f"simulations/products_{products}/factories_{factories}/stock_{stock}/shop_{shop}"
    #open csv file
    df = pd.read_csv(f"{path}/test_results.csv")

    print(f"({products},{factories},{stock},{shop})")
    #find mean value in column mean_objective
    print("HETEROSCEDASTICITY P VALUE:", df['heteroscedasticity_p_value'].mean())
    print("\n")



def draw_heat_map():
    #
    products = 1
    factories = 10
    stock = 10
    shop = 1
    #open csv file
    df = pd.read_csv(f"simulations/products_{products}/factories_{factories}/stock_{stock}/shop_{shop}/test_results.csv")

    #create df with columns confidence_level_shop and confidence_level_stock and integral_loss
    heatmap = df.pivot(index="confidence_level_shop", columns="confidence_level_stock", values="integral_loss") 

    #exp scale
    

    #import viridis color map
    cmap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256) 
    

    #make it rgb
    sns.heatmap(heatmap, cmap=cmap)
    plt.xlabel("beta")
    plt.ylabel("alpha")
    

    #save
    plt.savefig("heatmap.png")

find_optimal_level()