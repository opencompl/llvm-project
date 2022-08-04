import matplotlib.pyplot as plt

op = ["Union", "Intersect", "Subtract", "Complement", "IsEqual", "IsEmpty"]
branch = ["64-bit", "arbitary precision"]
data = []

# 64-bit
f = open("fpl", "r")
for i in range(6):
    for i in range(6):
        next(f)
    line = f.readline().strip().split()
    data.append([float(line[1]) / 10e6])
    print(line)
f.close()

# transprecision
f = open("tpint", "r")
for i in range(6):
    for j in range(6):
        next(f)
    line = f.readline().strip().split()
    data[i].append(float(line[1]) / 10e6)
    print(line)
f.close()

# Export plots
for i in range(6):
    name = "Compare" + op[i] + ".png"
    print(name)
    plt.figure(i)
    plt.bar(branch, data[i])
    plt.ylabel('ms')
    plt.savefig(name)
    print(data[i])

