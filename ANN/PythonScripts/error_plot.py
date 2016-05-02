from matplotlib import pyplot as plt

epochs = []
errors = []
with open('error.csv','r') as f:
    headers = f.readline()
    for line in f:
        data = [int(val) for val in line.split(',')]
        epochs.append(data[0])
        errors.append(data[1])

plt.scatter(epochs,errors)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
