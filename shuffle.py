import random
lines = open('dermatology.txt').readlines()
random.shuffle(lines)
open('dermatology.txt', 'w').writelines(lines)