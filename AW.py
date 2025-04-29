import numpy as np
from itertools import combinations
import random
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import interp1d

class UtilityBasedAllocation:
    def __init__(self, cost_mode="avg", price_mode="avg"):
        self.cost_mode = cost_mode
        self.price_mode = price_mode

    def calculate_global_cost(self, utilities, column_index, agents):
        selected_agents = utilities[agents, column_index]
        if self.cost_mode == "avg":
            return np.mean(selected_agents)
        elif self.cost_mode == "max":
            return np.max(selected_agents)
        elif self.cost_mode == "min":
            return np.min(selected_agents)
        else:
            raise ValueError("Invalid cost_mode. Choose 'avg', 'max', or 'min'.")

    def calculate_item_price(self, utilities, column_index):
        all_agents = utilities[:, column_index]
        if self.price_mode == "avg":
            return np.mean(all_agents)
        elif self.price_mode == "max":
            return np.max(all_agents)
        elif self.price_mode == "min":
            return np.min(all_agents)
        else:
            raise ValueError("Invalid price_mode. Choose 'avg', 'max', or 'min'.")

def adjusted_winner(A, split, epsilon=1e-10, use_money=True, prices=0, conversion_rate=(1, 1)):
    """
    Run the Adjusted Winner algorithm with optional money redistribution.
    Returns:
    - Minimum d value (utility difference)
    - Final utility of Agent 1
    - Final utility of Agent 2
    """
    m = A.shape[1]
    W1, W2 = [], []

    # Step 1: Initial allocation
    for j in range(m):
        if A[0, j] >= A[1, j]:
            W1.append(j)
        else:
            W2.append(j)

    agent1 = np.sum(A[0, W1])
    agent2 = np.sum(A[1, W2])
    d = abs(agent1 - agent2)

    # Step 2: Allow transfers to minimize d
    if split:
        transfer_pool = W1.copy() if agent1 > agent2 else W2.copy()

        # Calculate ratios and sort items by smallest ratio
        ratios = []
        for j in transfer_pool:
            ratio = A[0, j] / (A[1, j] + epsilon) if agent1 > agent2 else A[1, j] / (A[0, j] + epsilon)
            ratios.append((j, ratio))
        
        # Sort by ratio (ascending order)
        ratios.sort(key=lambda x: x[1])

        # Transfer resources from smallest ratio to largest
        for j, _ in ratios:
            # Verify that j exists before removing
            if agent1 > agent2 and j in W1:
                W1.remove(j)
                W2.append(j)
            elif agent2 > agent1 and j in W2:
                W2.remove(j)
                W1.append(j)
            else:
                continue  # Skip if j is no longer valid

            # Recalculate utilities and check if d is reduced
            new_agent1 = np.sum(A[0, W1])
            new_agent2 = np.sum(A[1, W2])
            new_d = abs(new_agent1 - new_agent2)

            if new_d <= d:  # Accept the transfer if it improves or maintains d
                d = new_d
                agent1, agent2 = new_agent1, new_agent2
            else:  # Undo the transfer if it doesn't improve d
                if agent1 > agent2:
                    W2.remove(j)
                    W1.append(j)
                elif agent2 > agent1:
                    W1.remove(j)
                    W2.append(j)
                break  # Exit the loop early as no further improvement is possible

    # Step 3: Use money redistribution to minimize d
    if use_money and prices > 0:
        frac = max(0, min(1, (prices + agent2 - agent1) / (2 * prices)))
        agent1 += frac * prices
        agent2 += (1 - frac) * prices
        d = abs(agent1 - agent2)

    return d, agent1, agent2



def check(A, costs, budget, split, use_money=True, prices=None):
    m = A.shape[1]
    results = []
    for num_deleted in range(m + 1):
        for deleted_columns in combinations(range(m), num_deleted):
            if np.sum(costs[list(deleted_columns)]) <= budget:
                modified_A = np.delete(A, deleted_columns, axis=1)
                revenue = np.sum(prices[list(deleted_columns)]) if prices is not None else 0
                d, agent1, agent2 = adjusted_winner(modified_A, split, use_money=use_money, prices=revenue)
                ratio = min(agent1 / agent2 if agent2 > 0 else 0,
                            agent2 / agent1 if agent1 > 0 else 0)
                results.append((d, ratio, len(deleted_columns)))
    if results:
        return min(results, key=lambda x: x[0])
    else:
        return float('inf'), 0, 0

class UtilityMatrixProcessor:
    @staticmethod
    def process_files(file_list, utility_model, m_range, budget_range):
        summary = defaultdict(lambda: defaultdict(list))
        for file_path in file_list:
            utilities = UtilityMatrixProcessor.extract_matrix(file_path)
            for m in m_range:
                if utilities.shape[1] == m and utilities.shape[0] >= 2:
                    agents = random.sample(range(utilities.shape[0]), 2)
                    reduced_utilities = utilities[agents, :]
                    costs = np.array([utility_model.calculate_global_cost(utilities, j, agents) for j in range(m)])
                    prices = np.array([utility_model.calculate_item_price(utilities, j) for j in range(m)])

                    for budget in budget_range:
                        d_nt, ratio_nt, sold_nt = check(reduced_utilities, costs, budget, split=False, use_money=True, prices=prices)
                        d_ns, ratio_ns, sold_ns = check(reduced_utilities, costs, budget, split=True, use_money=True, prices=prices)

                        summary[(utility_model.cost_mode, utility_model.price_mode, budget)]["AWNT"].append((d_nt, ratio_nt, sold_nt))
                        summary[(utility_model.cost_mode, utility_model.price_mode, budget)]["AWNS"].append((d_ns, ratio_ns, sold_ns))

        return summary

    @staticmethod
    def extract_matrix(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        first_number = int(lines[0].split()[0])
        matrix_data = [line.split() for line in lines[2:2 + first_number]]
        return np.array([[int(element) for element in row] for row in matrix_data])

if __name__ == "__main__":
    directory_path = r'C:\Users\User\Downloads\spliddit'
    file_pattern = directory_path + r'\*_*.INSTANCE'
    file_list = glob.glob(file_pattern)
    if not file_list:
        print("No files found!")
        exit()

    # All combinations of cost and price modes
    modes = [("avg", "avg"), ("max", "max"), ("min", "min"),
             ("avg", "max"), ("max", "avg"), ("min", "max"),
             ("max", "min"), ("avg", "min"), ("min", "avg")]

    m_range = range(4, 11)  # Utility matrix sizes
    budget_range = np.arange(0, 1001, 100)  # Budget values

    for cost_mode, price_mode in modes:
        utility_model = UtilityBasedAllocation(cost_mode=cost_mode, price_mode=price_mode)
        summary = UtilityMatrixProcessor.process_files(file_list, utility_model, m_range, budget_range)

        budgets = []
        avg_sold_awnt, avg_sold_awns = [], []
        avg_ratio_awnt, avg_ratio_awns = [], []

        for budget in budget_range:
            results = summary[(cost_mode, price_mode, budget)]
            awnt = results["AWNT"]
            awns = results["AWNS"]
            budgets.append(budget)

            avg_sold_awnt.append(np.mean([sold for _, _, sold in awnt]) if awnt else 0)
            avg_sold_awns.append(np.mean([sold for _, _, sold in awns]) if awns else 0)
            avg_ratio_awnt.append(1 - np.mean([ratio for _, ratio, _ in awnt]) if awnt else 0)
            avg_ratio_awns.append(1 - np.mean([ratio for _, ratio, _ in awns]) if awns else 0)

        threshold = 0.15


        # Interpolation for AWNT
        f_awnt = interp1d(avg_ratio_awnt, budgets, kind='linear', fill_value="extrapolate")
        budget_threshold_awnt = f_awnt(threshold)

        # Interpolation for AWNS
        f_awns = interp1d(avg_ratio_awns, budgets, kind='linear', fill_value="extrapolate")
        budget_threshold_awns = f_awns(threshold)

        # Plot Graphs
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Top Graph: Average Sold Items
        axes[0].plot(budgets, avg_sold_awnt, color="black", linestyle="-", label="AWNT")  # Solid line for AWNT
        axes[0].plot(budgets, avg_sold_awns, color="black", linestyle="--", label="AWNS")  # Dashed line for AWNS
        if budget_threshold_awnt is not None:
            axes[0].axvline(budget_threshold_awnt, color="red", linestyle="-", label="1 - Ratio = 15% (AWNT)")  # Red line
        if budget_threshold_awns is not None:
            axes[0].axvline(budget_threshold_awns, color="blue", linestyle="--", label="1 - Ratio = 15% (AWNS)")  # Blue dashed line
        axes[0].set_ylabel("Average Sold Items", fontsize=16)
        axes[0].set_ylim(0, 5)  # Set y-axis scale from 0 to 2
        #axes[0].legend(fontsize=12)

        # Bottom Graph: 1 - Average Ratio
        axes[1].plot(budgets, avg_ratio_awnt, color="black", linestyle="-", label="AWNT")  # Solid line for AWNT
        axes[1].plot(budgets, avg_ratio_awns, color="black", linestyle="--", label="AWNS")  # Dashed line for AWNS
        if budget_threshold_awnt is not None:
            axes[1].axvline(budget_threshold_awnt, color="red", linestyle="-", label="1 - Ratio = 15% (AWNT)")  # Red line
        if budget_threshold_awns is not None:
            axes[1].axvline(budget_threshold_awns, color="blue", linestyle="--", label="1 - Ratio = 15% (AWNS)")  # Blue dashed line
        axes[1].set_xlabel("Budget", fontsize=16)
        axes[1].set_ylabel("1 - Average Ratio", fontsize=16)
        axes[1].set_ylim(0, 0.5)  # Set y-axis scale from 0 to 0.5
        #axes[1].legend(fontsize=12)

        # Add title and layout
        plt.suptitle(f"Cost Mode: {cost_mode.capitalize()}, Price Mode: {price_mode.capitalize()}", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Show the figure
        plt.show()
