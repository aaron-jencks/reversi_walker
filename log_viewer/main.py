import csv
import matplotlib.pyplot as plt
import numpy as np

# headers = "runtime", "fps", "found", "explored", "disk_usage", "hashtable_load_factor"

if __name__ == '__main__':
    runtime = []
    fps = []
    disk = []
    load_factor = []
    with open(input('Which file would you like to open? ')) as fp:
        data = csv.reader(fp, delimiter=',')
        placeholder = 0
        previous = 0
        for i, t in enumerate(data):
            if i > 0:
                r, f, _, _, u, l = t
                r = int(r)
                f = int(f)
                u = float(u)
                l = float(l)
                if r <= previous:
                    placeholder = previous
                r += placeholder
                if f < 10000000:
                    runtime.append(r)
                    fps.append(f)
                    disk.append(u)
                    load_factor.append(l)
                previous = r

    fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].scatter(runtime, fps, 1)
    axs[0].set_ylabel('FPS (boards/sec)')

    axs[1].scatter(runtime, disk, 1)
    axs[1].set_ylabel('Disk Usage (%)')

    axs[2].scatter(runtime, load_factor, 1)
    axs[2].set_ylabel('Hashtable Load Factor')
    axs[2].set_xlabel('Runtime (sec)')

    plt.show()
