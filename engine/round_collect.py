import tqdm, chess, random

DIR = '/mnt/c/Users/undec/Desktop/combined/'
FILES = [
    ('split_dataset_over-0.csv', 2_000_000),
    ('split_dataset_over+200.csv', 2_000_000),
    ('split_dataset_over-200.csv', 2_000_000),
    ('split_dataset_over+400.csv', 2_000_000),
    ('split_dataset_over-400.csv', 2_000_000),
    ('split_dataset_over+600.csv', 2_000_000),
    ('split_dataset_over-600.csv', 2_000_000),
    ('split_dataset_over+800.csv', 2_000_000),
    ('split_dataset_over-800.csv', 2_000_000),
    ('split_dataset_over-1000.csv', 1_000_000),
    ('split_dataset_over+1000.csv', 1_000_000),
]

pos = []
for i in range(len(FILES)):
    poss = []
    with open(DIR + FILES[i][0], 'r') as fp:
        print('reading', FILES[i][0])
        fp.readline()
        while True:
            line = fp.readline()
            if line == '': break
            cr = line.split(' ')[2]
            valid = True
            for c in cr:
                if c not in 'KQkq':
                    valid = False
            if cr == '-': valid = True
            if valid: poss += [ line ]
            if len(poss) % 1000000 == 0: print(len(poss) // 1000000, 'M')
        print('Done reading', FILES[i][0])
    print('sampling...')
    pos += random.sample(poss, FILES[i][1])
    print('Done sampling.')

print('shuffling')
random.shuffle(pos)

print('writing')
with open('ds.csv', 'w') as fp:
    fp.write('fen,cp,mate\n')
    for line in pos:
        fp.write(line)
