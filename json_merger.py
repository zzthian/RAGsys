import json

with open("Simulator/data/tasks_kx_58.json", "r", encoding="utf-8") as f1, open("Simulator/data/tasks_kx_50.json", "r", encoding="utf-8") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# Find the max key in file1 (convert keys to int)
max_key = max(map(int, data1.keys()))

# Shift keys in file2
shifted_data2 = {}
for i, (k, v) in enumerate(data2.items(), start=1):
    new_key = str(max_key + i)
    shifted_data2[new_key] = v

# Merge both
merged = {**data1, **shifted_data2}

with open("Simulator/data/tasks_kx_108.json", "w") as f:
    json.dump(merged, f, indent=2)
