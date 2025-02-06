import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import time
from metrics import cal_rmse, cal_cost, calDTW, calAccuracy ,calDTW_new #评估指标

def time2ts(seq, time_scale):
    ts_list = []
    for t in list(seq):
        timeArray = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
        timeStamp = int(timeArray.timestamp()) * time_scale
        ts_list.append(timeStamp)
    return ts_list


def mode_interval_granularity(value):
    counter = Counter(value)
    sorted_scores = sorted(counter.items(),key=lambda x:x[1],reverse=True)
    return sorted_scores


def metric_res(truth_factors, repair, truth, fault, metric_name="cost", starts=[0]):
    """
    :param metric_id: 0: repair cost metric, 1: dtw metric, 2:rmse metric
    :return: loss
    """
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


def move(t,i,j,interval,s0,mt): #移动
    t_len = t[j] - t[s0]
    s_len = i * interval
    m = abs(t_len - s_len)/interval
    if m == 0:
        return mt
    else:
        return  (-1) * m
    
def get_sm(truth):
    eps_t_t = []
    s_0_t = []
    m_t = []
    s_0_t.append(truth[0])
    eps_t_t.append(truth[1]-truth[0])
    t = truth[1]-truth[0]
    m = 1
    for i in range(1,len(truth)):
        if (truth[i]-truth[i-1]) != t and i != (len(truth)-1):
            s_0_t.append(truth[i])
            t = truth[i+1] - truth[i]
            eps_t_t.append(t)
            m_t.append(m)
            m = 1
        else:
            m = m + 1
    return eps_t_t, s_0_t, m_t

def score_matrix(eps_t,t,lmd_a, lmd_d, mate, interval_granularity, k= 20 ):
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

def appr_repair(t, lmd_a, lmd_d, mate, interval_granularity):
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
    s_repair, eps_t_e, s_0_e, m_e = section(t,all_matrix,all_start,interval_list,interval_granularity)
    return s_repair, eps_t_e, s_0_e, m_e

def section(t, matrix, start, interval_list, interval_granularity):
    s_0_e = []
    eps_t_e = []
    m_e = []
    j=len(t)-1
    s_repair = []
    if type(matrix) is list:
        matrix = np.array(matrix)
    while(j>=0):
        index = np.argmax(matrix[:,j])
        j = start[index][j]
        s_0_e.insert(0,j)
        m_e.insert(0,interval_list[index][1]+1)
        eps_t_e.insert(0,interval_list[index][0])
        s_list = [round(t[j] + i * interval_list[index][0],interval_granularity) for i in range(interval_list[index][1]+1)]
        j=j-1
        s_list = [t for t in s_list if t < s_repair[0]]
        s_repair = s_list + s_repair
    return s_repair,eps_t_e, s_0_e, m_e

                     

if __name__ == "__main__":
    parameters = {
        "s-pm":{
            "file_counts": 5,
            "truth_col": 0,
            "truth_dir": "./data/pm",
            "original_col": 1,
            "original_dir": "./data/pm",
            "start_point_granularity": 1,
            "interval_granularity":36,
            "lmd_a": 36,
            "lmd_d": 36,
            "m_mate": 36,
        },
        "s-energy":{
            "file_counts": 5,
            "truth_col": 0,
            "truth_dir": "./data/energy",
            "original_col": 1,
            "original_dir": "./data/energy",
            "start_point_granularity": 1,
            "interval_granularity": 60,
            "lmd_a": 60,
            "lmd_d": 60,
            "m_mate": 60,
        },
        
    }

    version = "-test"
    datasets = ["s-pm","s-energy"]
    metrics = ["dtw"]
    methods = ["appr"]
    data_characteristic = False
    result_dfs = {}
    for m in metrics:
        result_dfs[m] = pd.DataFrame(0, columns=methods, index=datasets)
        result_dfs[m] = result_dfs[m].astype("float32")
    result_dfs["time"] = pd.DataFrame(0, columns=methods, index=datasets)
    result_dfs["time"] = result_dfs["time"].astype("float32")
    for dataset in datasets:
        param = parameters[dataset]
        file_counts = param["file_counts"]
        result_map = {}
        for method in methods:
            for metric in metrics:
                result_map[f'{method}-{metric}'] = []
            result_map[f'{method}-time'] = []
        dataset_path = os.path.join("./result", dataset)
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        for ts in range(file_counts):
            print(ts)
            original_dir = param["original_dir"]                
            file_name = os.path.join(original_dir, f"series_{ts}.csv")              
            data = pd.read_csv(file_name)
            original_seq = data.iloc[:, param["original_col"]]
            truth_dir = param["truth_dir"]              
            data_truth = pd.read_csv(os.path.join(truth_dir, f"series_{ts}.csv"))
            ground_truth_seq = data_truth.iloc[:, param["truth_col"]]
            
            if "time_scale" in param:
                original = time2ts(original_seq, param["time_scale"])
                truth = time2ts(ground_truth_seq, param["time_scale"])
            else:
                original = original_seq.dropna().tolist()
                truth = ground_truth_seq.dropna().tolist()
            eps_t_t, s_0_t, m_t = get_sm(truth)
            lmd_a = param["lmd_a"]
            lmd_d = param["lmd_d"]
            interval_granularity = int(param["interval_granularity"])
            mate = param["m_mate"]
            start = time.time()
            appr_res, eps_t_e, s_0_e, m_e = appr_repair(original, lmd_a, lmd_d, mate, interval_granularity)
            end = time.time()
            appr_time = end - start

            for metric in metrics:
                result_map[f"appr-{metric}"].append(metric_res([eps_t_t, s_0_t, m_t], appr_res, truth, original, metric))
            result_map[f"appr-time"].append(appr_time)              
        for metric in (metrics + ["time"]):
            result_dfs[metric].at[dataset, "appr"] = np.mean(result_map[f"appr-{metric}"])
            np.savetxt(os.path.join(dataset_path, f"appr-{metric}{version}.txt"), result_map[f"appr-{metric}"])                
    for metric in (metrics + ["time"]):
        result_dfs[metric].to_csv(os.path.join("result", f"exp1-{dataset}-{metric}{version}.csv"))


            







