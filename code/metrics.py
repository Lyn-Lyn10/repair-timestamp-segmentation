import math
from collections import defaultdict
import numpy as np
import pandas as pd
from dtw import dtw

def cal_rmse(truth_factors, truth, repair):
    ans = 0
    if isinstance(truth_factors[0], list):
        for i in range(len(truth_factors[0])):
            s_0 = truth_factors[1][i]
            if (i+1) == len(truth_factors[0]):
                s_end = truth[len(truth)-1] + 1
            else:
                s_end = truth[truth.index[truth == truth_factors[1][i+1]][0]-1]
            t = [x for x in truth if s_0 <= x <= s_end]
            r = [y for y in repair if s_0 <= y <= s_end]
            t = pd.Series(t)
            r = pd.Series(r)
            min_len = min(len(t), len(r))
            if min_len == 0:
                continue
            t, r = t[:min_len], r[:min_len]
            diff = abs(t - r)
            diff = diff.map(lambda x:math.pow(x,2))
            res = math.sqrt(sum(diff) / len(diff))
            ans = ans + res
    else:
        min_len = min(len(truth), len(repair))
        print(min_len)
        truth, repair = truth[:min_len], repair[:min_len]
        diff = abs(truth - repair)
        diff = diff.map(lambda x:math.pow(x,2))
        ans = math.sqrt(sum(diff) / len(diff))
    return ans


def cal_rmse_seg(truth, repair, starts):
    s = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    print("s:",s)
    s0 = starts[0]
    for i in range(len(s)):
        truth, repair = truth[s0:s0+s[i]], repair[s0:s0+s[i]]
        diff = [abs(truth[i] - repair[i]) for i in range(len(truth))]
        s0 = s0+s[i]
    return s


def calAccuracy(truth, fault, repair):
    min_len = min(len(truth), len(fault), len(repair))
    truth, fault, repair = truth[:min_len], fault[:min_len], repair[:min_len]
    error = sum(abs(truth - repair))
    cost = sum(abs(fault - repair))
    inject = sum(abs(truth - fault))
    if error == 0:
        return 1
    return (1 - (error / (cost + inject)))


def calDTW(truth, repair):
    distance, _, _, _ = dtw(truth, repair, dist=lambda x, y: np.linalg.norm(x - y))
    return distance


def cal_cost(truth, repair, lmd_a=5, lmd_d=5):
    s1 = repair
    s2 = truth
    n = len(s1)
    m = len(s2)
    dp = [[] for _ in range(n + 1)]
    dp[0].append(0)

    lmd_a = lmd_a * (truth[1] - truth[0])
    lmd_d = lmd_d * (truth[1] - truth[0])
    for i in range(1,n+1):
        dp[i].append(i*lmd_d)

    for j in range(1,m+1):
        dp[0].append(j*lmd_a)
        for i in range(1,n+1):
            s_m = s2[j-1]
            move_res = dp[i-1][j-1] + abs(s1[i-1]-s_m)
            add_res = dp[i][j - 1] + lmd_a
            del_res = dp[i-1][j]+lmd_d
            if move_res <= add_res and move_res <= del_res:
                dp[i].append(move_res)
            elif add_res <= move_res and add_res <= del_res:
                dp[i].append(add_res)
            else:
                dp[i].append(del_res)

    return dp[n][m]


