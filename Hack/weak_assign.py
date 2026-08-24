from sys import argv
import numpy as np

Ngr = int(argv[1])
Npr = 12
Nch = 4

char = lambda i: chr(65+i)

rng = np.random.default_rng(9)

for gr in range(Ngr) :
    choice = rng.choice(1+np.arange(Npr), size=Nch, replace=False)
    print(f'Group {char(gr)}: {choice}')
