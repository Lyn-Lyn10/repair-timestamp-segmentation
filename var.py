# var_Baseline
import pandas as pd
import numpy as np
import time
from dtw import dtw

def solve_quad(A, B, C):
    delta = B ** 2 - 4 * A * C  
    if delta <= 0: 
        x = B / (-2 * A) 
        return x, x
    else:  
        x1 = (B + delta ** 0.5) / (-2 * A) 
        x2 = (B - delta ** 0.5) / (-2 * A) 
        return x1, x2

def calDTW(truth, repair):
    distance, _, _, _ = dtw(truth, repair, dist=lambda x, y: np.linalg.norm(x - y))
    return distance

def variance_constraint_clean(df: pd.DataFrame, variance_constraints: dict, ra: list, t_attr, w, beta):
    attr = ra[0]
    sequence = np.array(df[attr])
    
    print(np.var(sequence[: w]))
    assert np.var(sequence[: w]) <= variance_constraints[attr]
        
    for k in range(w-1, df.shape[0] - 2*w):
        repair_sum = 0
        weight_sum = 0
        for i in range(k - w + 1, k + 1):
            num = i - (k - w + 1)
            weight_sum += pow(beta, num)
            if np.var(sequence[i: i+w]) <= variance_constraints[attr]:
                repair_sum += pow(beta, num) * sequence[k]
            else:
                l1_sum = np.sum(sequence[i: i+w]) - sequence[k]
                l2_sum = np.sum(sequence[i: i+w] * sequence[i: i+w]) - sequence[k] * sequence[k]
                x1, x2 = solve_quad(w-1, -2*l1_sum, w*l2_sum - l1_sum*l1_sum - w*w*variance_constraints[attr])
                    
                if x1 is None:
                    continue
                elif abs(x1 - sequence[k]) > abs(x2 - sequence[k]):
                    repair_sum += pow(beta, num) * x2
                else:
                    repair_sum += pow(beta, num) * x1

        repair = repair_sum / weight_sum
        if sequence[k] != repair:
            sequence[k] = repair

    df[attr] = sequence
    
    return df

if __name__ == '__main__':
    df = pd.read_csv('data/pm_seg.csv') 
    w = 10  
    column_name = 'dirty'  
    dff = df[[column_name]]
    variance_constraints = {
        'dirty': 10300,  
    }

    ra = ['dirty']
    t_attr = 'dirty'
    start_time = time.time()
    repaired_df = variance_constraint_clean(dff, variance_constraints, ra, t_attr, w, beta=0.5)
    repaired_column_list = repaired_df.iloc[:, 0].tolist()
    truth = df['truth'].tolist() 
    min_length = min(len(truth), len(repaired_column_list))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time: {elapsed_time:.4f} s")
    repaired_column_list = repaired_column_list[:min_length]
    truth = truth[:min_length]
    distance = calDTW(truth, repaired_column_list)
    print(f"DTW distance: {distance}")  


