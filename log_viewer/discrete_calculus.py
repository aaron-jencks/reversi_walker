
def first_derivative(data: list, time: list) -> list:
    return [((data[t + 1] - data[t - 1]) / (time[t] - time[t - 1] + (time[t + 1] - time[1])), time[t])
            for t in range(1, len(data) - 1, 1)]
