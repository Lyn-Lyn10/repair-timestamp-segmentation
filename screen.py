# SCREEN_Baseline
import time
from collections import deque
import numpy as np

class TimePoint:
    def __init__(self, timestamp, value):
        self.timestamp = timestamp
        self.value = value
        self.modify = value
        self.is_modified = False

    def set_modify(self, modify):
        self.modify = modify


class TimeSeries:
    def __init__(self):
        self.timeseries = []

    def add_time_point(self, time_point):
        self.timeseries.append(time_point)

    def get_timeseries(self):
        return self.timeseries

    def __iter__(self):
        return iter(self.timeseries)


class Screen:
    def __init__(self, timeseries, s_max, s_min, t):
        self.timeseries = timeseries
        self.s_max = s_max
        self.s_min = s_min
        self.t = t
        self.kp = None

    def main_screen(self):
        total_list = self.timeseries.get_timeseries()
        size = len(total_list)

        pre_end = -1
        temp_series = TimeSeries()
        read_index = 1

        tp = total_list[0]
        temp_series.add_time_point(tp)
        w_start_time = tp.timestamp
        w_end_time = w_start_time
        w_goal_time = w_start_time + self.t

        while read_index < size:
            tp = total_list[read_index]
            cur_time = tp.timestamp

            if cur_time > w_goal_time:
                while True:
                    temp_list = temp_series.get_timeseries()
                    if not temp_list:
                        temp_series.add_time_point(tp)
                        w_goal_time = cur_time + self.t
                        w_end_time = cur_time
                        break

                    self.kp = temp_list[0]
                    w_start_time = self.kp.timestamp
                    w_goal_time = w_start_time + self.t

                    if cur_time <= w_goal_time:
                        temp_series.add_time_point(tp)
                        w_end_time = cur_time
                        break

                    cur_end = w_end_time

                    if pre_end == -1:
                        pre_point = self.kp

                    self.local(temp_series, pre_point)

                    pre_point = self.kp
                    pre_point.is_modified = True
                    pre_end = cur_end

                    temp_series.get_timeseries().pop(0)
            else:
                if cur_time > w_end_time:
                    temp_series.add_time_point(tp)
                    w_end_time = cur_time

            read_index += 1

        result_series = TimeSeries()
        for time_point in self.timeseries.get_timeseries():
            result_series.add_time_point(TimePoint(time_point.timestamp, time_point.modify))

        return result_series

    def local(self, time_series, pre_point):
        temp_list = time_series.get_timeseries()
        pre_time = pre_point.timestamp
        pre_val = pre_point.modify
        kp_time = self.kp.timestamp

        lower_bound = pre_val + self.s_min * (kp_time - pre_time)
        upper_bound = pre_val + self.s_max * (kp_time - pre_time)

        xk_list = [self.kp.modify]
        for tp in temp_list[1:]:
            val = tp.modify
            d_time = kp_time - tp.timestamp
            xk_list.extend([val + self.s_min * d_time, val + self.s_max * d_time])

        xk_list.sort()
        x_mid = xk_list[len(temp_list) - 1]
        modify = min(max(x_mid, lower_bound), upper_bound)

        self.kp.set_modify(modify)


class Assist:
    def read_data(self, filename, index, split_op):
        time_series = TimeSeries()
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split(split_op)
                timestamp, value = int(parts[0]), float(parts[index])
                time_series.add_time_point(TimePoint(timestamp, value))
        return time_series

    def calc_rms(self, truth_series, test_series):
        truth_list = [tp.modify for tp in truth_series]
        test_list = [tp.modify for tp in test_series]
        return (sum((t - s) ** 2 for t, s in zip(truth_list, test_list)) / len(truth_list)) ** 0.5
    
    def calculateDTW(self, series1, series2):
        n, m = len(series1), len(series2)
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(series1[i - 1] - series2[j - 1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
        
        return dtw_matrix[n, m]


def main():
    input_file = "data/pm/pm.data"
    start_time = time.time()
    assist = Assist()

    dirty_series = assist.read_data(input_file, 1, ",")
    truth_series = assist.read_data(input_file, 0, ",")

    s_max, s_min, t = 100, -100, 100
    screen = Screen(dirty_series, s_max, s_min, t)
    result_series = screen.main_screen()

    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    print(f"Time:{execution_time:.2f} ms")

    result_list = [tp.modify for tp in result_series]
    truth_list = [tp.modify for tp in truth_series]

    repair_dtw = assist.calculateDTW(truth_list, result_list)
    print(f"Repair DTW distance is {repair_dtw:.2f}")


if __name__ == "__main__":
    main()
