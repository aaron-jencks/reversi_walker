import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
import sys

from discrete_calculus import first_derivative

if __name__ == '__main__':
    fig = plt.figure()
    first = True

    for fi, f in enumerate(sys.argv[1:]):
        headers = []
        rdata = []
        fname = f.rsplit('/', 1)[1]

        with open(f) as fp:
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

        start_index = fi + 1
        first_axs = True
        prev = None

        for a, y in enumerate(rdata[1:]):
            ax_index = a * len(sys.argv[1:])

            if first_axs:
                axs = fig.add_subplot(len(headers) - 1, len(sys.argv[1:]), start_index + ax_index)
                axs.set_title(fname)
                prev = axs
                first_axs = False
            else:
                axs = fig.add_subplot(len(headers) - 1, len(sys.argv[1:]), start_index + ax_index)

            axs.scatter(rdata[0], y, 1)

            ylbl = str(headers[a + 1])

            if ylbl == 'fps':
                axs.text(max(rdata[0]) - (2 * max(rdata[0]) / 3), max(y) / 2,
                         'Average FPS: {} b/s'.format(round(stat.mean(y), 4)))

            if ylbl == 'collisions':
                der = first_derivative(y, rdata[0])
                dc = [d[0] for d in der]
                dt = [d[1] for d in der]
                axs.scatter(dt, dc, 1, c=[[0.5, 0, 0.5, 0.8]])
                axs.text(max(rdata[0]) - (max(rdata[0]) / 2), max(y) / 4, 'Average Slope: {} b/s'.format(round(stat.mean(dc), 4)))

            if '_' in ylbl:
                temp = ''
                words = ylbl.split('_')
                for w in words:
                    temp += w[0]
                ylbl = temp

            axs.set_ylabel(ylbl)
            if a == len(headers) - 2:
                axs.set_xlabel('Runtime (sec)')
            else:
                axs.set_xticks([])

    plt.show()
