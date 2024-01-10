import random


def parse_vector(arr):
    num = 0
    arr.reverse()
    for i in range(len(arr)):
        if arr[i] == 1:
            num = i + 1
    return num
