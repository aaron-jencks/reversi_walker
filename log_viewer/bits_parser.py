import struct
from typing import List, Iterable

from tqdm import tqdm
import matplotlib.pyplot as plt


def parse_bits_file(filename: str, page_size: int) -> Iterable[List[int]]:
    with open(filename, mode='rb') as fp:
        data = fp.read()

        pbar = tqdm(total=len(data))
        pointer = 0
        while pointer < len(data):
            result = []
            ukey, lkey = struct.unpack('@QQ', data[pointer:pointer + 16])
            key = (lkey << 128) + ukey
            if key == 0:
                pbar.close()
                yield None
                break
            for bi in range(page_size - 16):
                b = data[pointer + 16 + bi]
                key_base = key + bi
                for bit in range(8):
                    if b & (1 << bit):
                        ikey = key_base + bit
                        result.append(ikey)
            yield result
            pbar.update(page_size)
        pbar.close()


if __name__ == '__main__':
    bins = [0 for _ in range(1000)]

    for b in parse_bits_file('/home/aaron/Temp/reversi.VwoK79/bits/p0.bin', 272):
        for k in b:
            bin = k % 1000
            bins[bin] += 1

    plt.scatter(list(range(1000)), bins)
    plt.show()
