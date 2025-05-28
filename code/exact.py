import os
import pandas as pd
import numpy as np
from collections import Counter

def determine_interval(t):
    eps = [t[i] - t[i-1] for i in range(1,len(t-1))]
    return np.median(eps)

def match_searching(t, eps_t, s_0, d, lmd_a, lmd_d, lmd_m, DP):
    n = len(t)
    ln = t[n-1] - t[0]
    dp = [[] for i in range(n+1-d)]
    op = [[] for i in range(n+1-d)]
    l = [0 for i in range(n+1)]
    dp[0].append(0)
    op[0].append(0)
    for i in range(1,n+1-d):
        dp[i].append(i*lmd_d)
        op[i].append(2)
    m = 1
    while m * eps_t <= (ln - s_0 + t[0]):
        dp[0].append(m*lmd_a)
        op[0].append(1)
        for i in range(1,n+1-d):
            s_m = s_0 + (m-1)*eps_t
            move_res = dp[i-1][m-1]+abs(t[i-1+d]-s_m)
            add_res = dp[i][m - 1] + lmd_a
            del_res = dp[i-1][m]+lmd_d
            if move_res <= add_res and move_res <= del_res:
                dp[i].append(move_res)
                op[i].append(0)
            elif add_res <= move_res and add_res <= del_res:
                dp[i].append(add_res)
                op[i].append(1)
            else:
                dp[i].append(del_res)
                op[i].append(2)
        for i in range(d,n+1):
                if DP[d][i][0] > dp[i-d][m]:
                    DP[d][i][0] = dp[i-d][m]
                    DP[d][i][1] = eps_t
                    DP[d][i][2] = m
        m += 1
    return DP


def mode_interval_granularity(value):
    counter = Counter(value)
    sorted_scores = sorted(counter.items(),key=lambda x:x[1],reverse=True)
    return sorted_scores

def round_to_granularity(value, granularity):
    return round(value / granularity) * granularity

def exact_repair(t, lmd_a, lmd_d, lmd_m, l,interval_granularity, start_point_granularity=1, bias_d=1, bias_s=1):
    eps_list = [t[i] - t[i - 1] for i in range(1, len(t) - 1)]
    eps_md = np.median(eps_list)
    interval = mode_interval_granularity(eps_list)
    n = len(t)
    DP = [[[10e4,0,0]for j in range(n+1)] for i in range(n)]
    eps_t = round_to_granularity(eps_md, interval_granularity) 
    for eps_t,num in interval[0:2]:
        if eps_t==0:
            continue
        d = 0
        while d!=n:
            s_0 = t[d] 
            DP = match_searching(t, eps_t, s_0, d, lmd_a, lmd_d, lmd_m, DP)
            d += 1
    s = check_M(DP,t,l)
    return s
    
def check_M(DP,t,l):
    n = len(t)
    M = [[10e8,[]] for i in range(n+2)]
    for i in range(2*l):
        M[i][0] = DP[0][i+1][0]
        M[i][1] = [t[0] + h*DP[0][i+1][1] for h in range(DP[0][i+1][2]+1)]
    for i in range(2*l,n):
        m = 10e8
        for jj in range(i-2*l+1):
            j = l-1+jj
            if M[j][0] + DP[j+2][i+1][0] < m:
                s_0 = t[j+2]
                eps_t = DP[j+2][i+1][1]
                k = DP[j+2][i+1][2]
                num = j
            m = min(m,M[j][0] + DP[j+2][i+1][0])
        M[i][0] = min(DP[0][i+1][0],m)    
        if DP[0][i+1][0] <= m :
            s_0 = t[0]
            M[i][1] = [s_0 + h*DP[0][i+1][1] for h in range(DP[0][i+1][2]+1)]
        else:
            M[i][1] = M[num][1]+[s_0 + h*eps_t for h in range(k+1)]
    
    return  M[n-1][1]














