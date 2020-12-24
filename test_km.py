import src.KM as KM
import json
import numpy as np
# from hungarian_algorithm import algorithm

# value = {}
# data_loc = json.load(open("loc_class0.json"))
# data_angle = json.load(open("class0.json"))

# value = []
# for i in range(len(data_loc)):
#     value.append(
#         (data_loc[i][0]["label"],
#          data_loc[i][1]["label"],
#          -np.linalg.norm(
#             np.concatenate(
#                 (np.array(data_loc[i][0]["pos"]), np.array(data_angle[i]["angle1"])))
#             - np.concatenate((np.array(data_loc[i][1]["pos"]),
#                               np.array(data_angle[i]["angle2"]).reshape(1,)))
#         )))

# value = [(d[0]["label"], d[1]["label"], -np.linalg.norm(
#     np.array(d[0]["pos"]) - np.array(d[1]["pos"]))) for d in data_loc]

# json.dump(value, open("value.json", mode="w"))

data = json.load(open("value.json"))
result = KM.run_kuhn_munkres(data)
cnt = 0
for d in result:
    if d[0] == d[1]:
        cnt = cnt + 1
print(cnt, cnt / len(result))
