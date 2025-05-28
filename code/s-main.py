import os
import numpy as np
import pandas as pd
from datetime import datetime
from exact import exact_repair 
from metrics import cal_rmse, cal_cost, calDTW, calAccuracy 
import time

def time2ts(seq, time_scale):
    ts_list = []
    for t in list(seq):
        timeArray = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
        timeStamp = int(timeArray.timestamp()) * time_scale
        ts_list.append(timeStamp)
    return ts_list

def equal_series_generate(eps_t, s_0, m):
    return [s_0 + i*eps_t for i in range(m)]

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

def metric_res(truth_factors, repair, truth, fault, metric_name="cost"):
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


if __name__ == "__main__":
    parameters = {
        "s-energy":{
            "file_counts": 5,
            "truth_col": 0,
            "truth_dir": "../data/energy",
            "original_col": 1,
            "original_dir": "../data/energy",
            "start_point_granularity": 1,
            "interval_granularity": 60,
            "lmd_a": 60,
            "lmd_d": 60,
            "lmd_m": 60,
            "l_min": 5,
        },
    }
    

    version = "-test"
    datasets = ["s-energy"]
    metrics = ["dtw"]
    methods = ["exact"]
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

        dataset_path = os.path.join("../result", dataset)
        print(os.path.abspath("./"))
        if not os.path.exists("../result"):
            os.mkdir("../result")
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        for ts in range(file_counts):
            print(f"Processing file #{ts}")
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
            lmd_m = param["lmd_m"]
            l = param["l_min"]
            interval_granularity = param["interval_granularity"]
            start_point_granularity = param["start_point_granularity"]
            start = time.time()
            exact_res = exact_repair(original, lmd_a, lmd_d, lmd_m, l, interval_granularity, start_point_granularity)
            print("exact end")
            end = time.time()
            exact_time = end - start
            for metric in metrics:
                result_map[f"exact-{metric}"].append(metric_res([eps_t_t, s_0_t, m_t], exact_res, truth, original,metric))
            result_map[f"exact-time"].append(exact_time)
        for metric in (metrics + ["time"]):
            result_dfs[metric].at[dataset, "exact"] = np.mean(result_map[f"exact-{metric}"])
            np.savetxt(os.path.join(dataset_path, f"exact-{metric}{version}.txt"), result_map[f"exact-{metric}"])
    for metric in (metrics + ["time"]):
        result_dfs[metric].to_csv(os.path.join("result", f"exp1-{metric}{version}.csv"))
