from itertools import permutations
data_path = 'data/data'
save_path = 'data'
filename = 'data'
def data_strengthen():
    tiles = 'WTB'
    tmps = ['__W__', '__T__', '__B__']
    for perm in permutations(tiles):
        perm = ''.join(perm)
        print ('Generate for ', perm)
        m = {}
        for x,y in zip(tmps, perm):
            m[x] = y
        with open(f'{data_path}/{filename}.txt', encoding='UTF-8') as f:
            data = f.read()
        data = data.replace('Wind', '__X__')
        data = data.replace('BuGang', '__Y__')
        for x,y in zip(tiles, tmps):
            data = data.replace(x, y)
        for x, y in zip(tmps, perm):
            data = data.replace(x, y)
        data = data.replace('__X__', 'Wind')
        data = data.replace('__Y__', 'BuGang')
        with open(f'{data_path}/{filename}-{perm}.txt', 'w', encoding='UTF-8') as f:
            f.write(data)
data_strengthen()