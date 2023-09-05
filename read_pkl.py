import pickle
import os

# path = './keypoint_results.pkl'
path = './mmdetection/results.pkl'

f = open(path, 'rb')

data = pickle.load(f)
print(len(data))

# print(data[0])

# for i in range(len(data)):
print(data[0])
# i = 0
# shoti = 0
# long = float("inf")
# for item in data:
#     if len(item) < long:
#         long = len(item)
#         shoti = i
#     i += 1
#     print(i)
# print(data[shoti])