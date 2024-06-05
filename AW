import numpy as np
from itertools import combinations
import glob


def adjusted_winner(A, split):
    m = A.shape[1]
    flag = False
    W1 = []
    W2 = []
    for j in range(m):
        if A[0, j] >= A[1, j]:
            W1.append(j)
        else:
            W2.append(j)

    agent1 = np.sum(A[0, W1])
    agent2 = np.sum(A[1, W2])

    flag = 0  
    d = abs(agent1 - agent2)
    
    transfer = [0]
    delete = [0]
    if not split:
        return d
    else:
        if agent1 > agent2:
            flag = 1
            non_zero_idx = np.nonzero(A[1] != 0)[0]
            ratios = np.divide(A[0, non_zero_idx], A[1, non_zero_idx])
        else:
            non_zero_idx = np.nonzero(A[0] != 0)[0]
            ratios = np.divide(A[1, non_zero_idx], A[0, non_zero_idx])
        

        filtered_indices = np.where(ratios >= 1)[0]    
        indices_ratios_geq_1 = non_zero_idx[filtered_indices]
        sorted_indices = np.argsort(ratios[filtered_indices])
        A_f = A[:, indices_ratios_geq_1[sorted_indices]]
        
        if len(sorted_indices) == 0:
            return d
        
        if flag == 1:
            for i in range(len(A_f[0])):
                diff = abs((agent1 - A_f[0][i] - sum(delete)) - (sum(transfer) + agent2 + A_f[1][i]))
                if diff < d:
                    d = diff
                    transfer.append(A_f[1][i])
                    delete.append(A_f[0][i])
        else:
            for i in range(len(A_f[0])):
                diff = abs((agent2 - A_f[1][i] - sum(delete)) - (sum(transfer) + agent1 + A_f[0][i]))
                if diff < d:
                    d = diff
                    transfer.append(A_f[0][i])
                    delete.append(A_f[1][i])
    return d

def check(A, k, booli):
    m = A.shape[1]
    d_values = [] 
    for j in range(k+1):
        for deleted_columns in combinations(range(m), j):
            modified_A = np.delete(A, deleted_columns, axis=1)
            d_values.append(adjusted_winner(modified_A, booli))
    return min(d_values)



def generate_central_approval_set(p, m):
    return np.random.choice(np.arange(m), size=int(p * m), replace=False)

def generate_agent_approval_set(central_set, phi, p):
    agent_approval_sets = []
    for good in central_set:
        if np.random.rand() < phi:
            central_set[good] = good if np.random.rand() < p else -1
            
    return central_set

def generate_dirichlet_resampling_utility_matrix(num_agents, num_goods, p, phi, alpha):
    utility_matrix = np.zeros((num_agents, num_goods), dtype=int) 

    for agent in range(num_agents):
        approval_set = np.random.choice(np.arange(num_goods), size=int(p * num_goods), replace=False)
        for good in range(num_goods):
            if good not in approval_set:
                utility_matrix[agent][good] = 0
            else:
                utility_matrix[agent][good] = int(np.round(np.random.dirichlet(np.ones(len(approval_set)) * alpha)[approval_set.tolist().index(good)] * 100))  # Scale by 100 and round

    return utility_matrix

def generate_euclidean_utility_matrix(num_agents, num_goods, dimension):
    utility_matrix = np.zeros((num_agents, num_goods), dtype=int)
    agent_vectors = np.random.rand(num_agents, dimension)
    good_vectors = np.random.rand(num_goods, dimension)
    for agent in range(num_agents):
        for good in range(num_goods):
            max_distance = np.max(np.linalg.norm(agent_vectors[agent] - good_vectors, axis=1))
            distance = np.linalg.norm(agent_vectors[agent] - good_vectors[good])
            utility_matrix[agent][good] = int(np.round((1 - (distance / max_distance)) * 100))  # Scale by 100 and round

    return utility_matrix

def generate_attributes_utility_matrix(num_agents, num_goods, dimension):
    utility_matrix = np.zeros((num_agents, num_goods), dtype=int) 
    goods_attributes = np.random.rand(num_goods, dimension)
    agent_utilities = np.random.rand(num_agents, dimension)
    
    for agent in range(num_agents):
        for good in range(num_goods):
            utility_matrix[agent][good] = int(np.round(np.dot(agent_utilities[agent], goods_attributes[good]) * 100))  # Scale by 100 and round

    return utility_matrix

def adjust_matrix(matrix):
    target_sum = 1000
    for i in range(matrix.shape[0]):
        row = matrix[i]
        current_sum = np.sum(row)
        if current_sum != 0:
            row_ratio = target_sum / current_sum
            row = np.round(row * row_ratio).astype(int)
            adjustment = target_sum - np.sum(row)
            if adjustment != 0:
                max_index = np.argmax(row)
                row[max_index] += adjustment
        matrix[i] = row
    return matrix

def extract_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    first_number = int(lines[0].split()[0]) 
    matrix_data = [line.split() for line in lines[2:2+first_number]]  
    matrix = np.array([[int(element) for element in row] for row in matrix_data])
    return matrix

def calculate_utility_matrices(num_agents, p_val, phi_val, alpha_val, dimension):
    dirichlet_resampling_utility = []
    attributes_utility = []
    euclidean_utility = []
    for s in range(0,5):                      
        for m1 in range(4, 11):
            for pval in p_val:
                for phival in phi_val:
                    for alphaval in alpha_val:
                        for dim in dimension:
                            dirichlet_resampling_utility.append(adjust_matrix(generate_dirichlet_resampling_utility_matrix(num_agents, m1, pval, phival, alphaval))) 
                            attributes_utility.append(adjust_matrix(generate_attributes_utility_matrix(num_agents, m1, dim))) 
                            euclidean_utility.append(adjust_matrix(generate_euclidean_utility_matrix(num_agents, m1, dim)))

    return dirichlet_resampling_utility, attributes_utility, euclidean_utility

def process_utility_matrices(num_goods, dirichlet_resampling_utility, attributes_utility, euclidean_utility):
    dicTdir={}
    dicSdir={}
    dicTatt={} 
    dicSatt={}
    dicTeuc={} 
    dicSeuc={}
    for num in range(4, 11):
        for k in range(num - 1):
            key = (num, k)
            dicTdir[key] = []
            dicSdir[key] = []
            dicTatt[key] = []
            dicSatt[key] = []
            dicTeuc[key] = []
            dicSeuc[key] = []
            for i in range(len(euclidean_utility)):
                if len(dirichlet_resampling_utility[i][0]) == num:   
                    dicTdir[key].append(check(dirichlet_resampling_utility[i], k, False))
                    dicSdir[key].append(check(dirichlet_resampling_utility[i], k, True)) 
                if len(attributes_utility[i][0]) == num:
                    dicTatt[key].append(check(attributes_utility[i], k, False))
                    dicSatt[key].append(check(attributes_utility[i], k, True))
                if len(euclidean_utility[i][0]) == num:
                    dicTeuc[key].append(check(euclidean_utility[i], k, False))
                    dicSeuc[key].append(check(euclidean_utility[i], k, True))

    return dicTdir, dicSdir, dicTatt, dicSatt, dicTeuc, dicSeuc

def process_file_list(file_list):
    dicTdata = {}
    dicSdata = {}
    for m in range(4, 11):
        for k in range(m - 1):
            key = (m, k)
            dicTdata[key] = []
            dicSdata[key] = []
            for file_path in file_list:
                preferences = np.array(extract_matrix(file_path))
                if len(preferences) == 2 and len(preferences[0]) == m:
                    dicTdata[key].append(check(preferences, k, False))
                    dicSdata[key].append(check(preferences, k, True))

    return dicTdata, dicSdata

if __name__ == "__main__":
    num_agents = 2
    p_val = [0.6,0.4,0.2]
    phi_val = [0.2,0.8]
    alpha_val = [1.0,2.0,3.0]
    dimension = [2,5]
    dic_arr = []
    directory_path = r'C:\Users\User\Downloads\spliddit'
    file_pattern = directory_path + r'\*_*.INSTANCE'
    file_list = glob.glob(file_pattern)

    dirichlet_resampling_utility, attributes_utility, euclidean_utility = calculate_utility_matrices(num_agents, p_val, phi_val, alpha_val, dimension)

    dicTdir, dicSdir, dicTatt, dicSatt, dicTeuc, dicSeuc = process_utility_matrices(num_agents, dirichlet_resampling_utility, attributes_utility, euclidean_utility)

    dicTdata, dicSdata = process_file_list(file_list)
