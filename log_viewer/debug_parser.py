def find_duplicate_hashes(file: str) -> tuple:
    result = {}
    repeats = {}
    with open(file, 'r') as fp:
        for line in fp.readlines():
            bits = line.split(' ')
            if len(bits) == 5:
                _, _, _, u, l = line.split(' ')
                n = (int(u) << 64) + int(l)
                if n not in result:
                    result[n] = 1
                else:
                    result[n] += 1
            else:
                _, _, _, u, l, _, _ = line.split(' ')
                n = (int(u) << 64) + int(l)
                if n not in repeats:
                    repeats[n] = 1
                else:
                    repeats[n] += 1
    return result, repeats
