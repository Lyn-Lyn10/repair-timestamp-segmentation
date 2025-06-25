import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
from metrics import cal_rmse, cal_cost, calDTW, calAccuracy 


def mode_interval_granularity(value):
    counter = Counter(value)
    sorted_scores = sorted(counter.items(),key=lambda x:x[1],reverse=True)
    return sorted_scores


def metric_res(truth_factors, repair, truth, fault, metric_name="cost", starts=[0]):
    if metric_name == "cost":
        lmd_a = 5 * (truth[1] - truth[0])
        lmd_d = 5 * (truth[1] - truth[0])
        return cal_cost(truth, repair, lmd_a, lmd_d)
    elif metric_name == "dtw":
        return calDTW(truth, repair)
    elif metric_name == "accuracy":
        truth = pd.Series(truth)
        repair = pd.Series(repair)
        return calAccuracy(truth, fault, repair)
    else:
        truth = pd.Series(truth)
        repair = pd.Series(repair)
        return cal_rmse(truth_factors, truth, repair)


def move(t, i, j, interval, s0, mt): 
    t_len = t[j] - t[s0]
    s_len = i * interval
    m = abs(t_len - s_len)/interval
    if m == 0:
        return mt
    else:
        return  (-1) * m
    

def score_matrix(eps_t, t, lmd_a, lmd_d, mate, interval_granularity, k=20):
    s_num = round((t[len(t) - 1] - t[0]) / eps_t + 2)
    dp =[[-100] * len(t) for i in range (s_num)]
    st = [[0] * len(t) for i in range(s_num) ]
    s_0 = []
    for j in range(len(t)):
        dp[0][j] = mate
        st[0][j] = j
        if j != 0:
            if abs(t[j] - t[j - 1]) > (k * eps_t):
                s_0.append(j)

    for i in range(1,s_num):
        for j in range(len(t)):
            dic = { 0 : st[i-1][j-1], 1 : st[i-1][j], 2 : st[i][j-1]}
            if j == 0:
                dp[i][0] =round(dp[i-1][0] - lmd_a,interval_granularity)
                continue
            else:
                if j in s_0:
                    dp[i][j] = round(dp[i-1][j] - lmd_a,interval_granularity)
                    st[i][j] = st[i-1][j]
                else:
                    m = round(move(t,i,j,eps_t,st[i-1][j-1],mate),interval_granularity)
                    a = round(dp[i-1][j] - lmd_a,interval_granularity)
                    d = round(dp[i][j-1] - lmd_d,interval_granularity)
                    arr = [round((dp[i-1][j-1] + m),interval_granularity),a,d]
                    dp[i][j] = round(max(arr),interval_granularity)
                    max_index = arr.index(dp[i][j]) 
                    st[i][j] = dic[max_index]
    return dp,st,s_num

def appr_repair(t, lmd_a, lmd_d, mate, interval_granularity, l_min):
    eps_list = [round(t[i] - t[i - 1],interval_granularity) for i in range(1, len(t))] 
    interval = mode_interval_granularity(eps_list)
    media = np.median(eps_list)
    if media == interval[0][0]:
        T_num = 1
    else:
        T_num = 3
    i = 0
    interval_list=[]
    for eps_t,num in interval[0:T_num]:
        if eps_t == 0:
            continue
        mt ,st ,m = score_matrix(eps_t,t,lmd_a,lmd_d,mate,interval_granularity)
        i = i+1
        for j in range(m):
            interval_list.append([eps_t,j])
        if i == 1:
            all_matrix = mt
            all_start = st
            continue
        else:
            all_matrix = np.concatenate((all_matrix,mt))
            all_start = np.concatenate((all_start,st))
    s_repair, eps_t_e, s_0_e, m_e = section(t, all_matrix, all_start, interval_list, interval_granularity, l_min)
    return s_repair, eps_t_e, s_0_e, m_e

def section(t, matrix, start, interval_list, interval_granularity, l_min):
    s_0_e = []
    eps_t_e = []
    m_e = []
    j=len(t)-1
    s_repair = []
    if type(matrix) is list:
        matrix = np.array(matrix)
    while(j>=0):
        index = np.argmax(matrix[:,j])
        if (np.max(matrix[:,j])==-10e8):
            print(start[index][j])
            print(interval_list[index][1]+1)
            print("Can not be repaired according to l_min requirements.")
            break
        elif ((interval_list[index][1]+1) >= l_min or start[index][j]==0):
            j = start[index][j]
            s_0_e.insert(0,j)
            m_e.insert(0,interval_list[index][1]+1)
            eps_t_e.insert(0,interval_list[index][0])
            s_list = [round(t[j] + i * interval_list[index][0],interval_granularity) for i in range(interval_list[index][1]+1)]
            j=j-1
            if s_repair:
                s_list = [t for t in s_list if t < s_repair[0]]
            s_repair = s_list + s_repair
        else:
            matrix[index,j] = -10e8
    return s_repair,eps_t_e, s_0_e, m_e
                