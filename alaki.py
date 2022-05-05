import os
import glob

path = 'exp_data/*/results/log/*.res'
names = [name for name in glob.glob(path)]
for name in names:
    txt_name = name.replace('.res', '.txt')
    os.rename(name, txt_name)