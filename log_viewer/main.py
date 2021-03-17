import csv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    headers = []
    rdata = []

    with open(input('Which file would you like to open? ')) as fp:
        data = csv.reader(fp, delimiter=',')
        placeholder = 0
        previous = 0
        for i, t in enumerate(data):
            if i == 0:
                headers = list(t)
                for _ in range(len(t)):
                    rdata.append([])
            else:
                for j, e in enumerate(t):
                    e = float(e)
                    if j == 0:
                        if e <= previous:
                            placeholder = previous
                        e += placeholder
                        previous = e
                    if j == 1 and e >= 10000000:
                        rdata[0].pop(len(rdata[0]) - 1)
                        break
                    rdata[j].append(e)

    fig, axs = plt.subplots(len(headers) - 1, 1, sharex=True)

    for a, y in enumerate(rdata[1:]):
        axs[a].scatter(rdata[0], y)
        axs[a].set_ylabel(headers[a + 1])

    axs[len(headers) - 2].set_xlabel('Runtime (sec)')

    plt.show()
