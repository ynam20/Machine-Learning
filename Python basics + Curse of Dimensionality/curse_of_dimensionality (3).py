import matplotlib.pyplot as plt
import numpy as np
import math
if __name__ == "__main__":
    # Python2 handles integer division differently from Python3.
    # The following line will cause Python2 to terminate, but let Python3 continue.
    assert 1/2 == .5, "Use Python3, not Python2"


def findratio(inputarray, num):
        min = np.linalg.norm(inputarray[0] - inputarray[1])
        max = np.linalg.norm(inputarray[0] - inputarray[1])
        for i in range(num):
            for j in range(i+1, num):
                    if (np.linalg.norm(inputarray[i] - inputarray[j]) > max):
                        max = np.linalg.norm(inputarray[i] - inputarray[j])
                    if (np.linalg.norm(inputarray[i] - inputarray[j]) < min):
                        min = np.linalg.norm(inputarray[i] - inputarray[j])
        ratio = min / max
        return ratio

def findedgeratio(inputarray, dimension):
        counter = 0

        for m in range(100):
            print("point counter", m)
            for n in range(dimension):
                print("m", m, "n", n, "dimension", dimension)

                if (inputarray[m][n]  < .01) or (inputarray[m][n] >= .99):
                    print(inputarray[m][n])
                    counter += 1
                    print("close to edge counter", counter)
                    print("i made it")
                    break
                if n == (dimension - 1):
                    print("I DIDNT MAKE IT")

                    break

        return counter/100

def findcircleratio(inputarray, dimension):
        counter = 0
        centerpoint = [0.5]*dimension

        for i in range(100):
            if ((np.linalg.norm(inputarray[i] - centerpoint) < 0.5)):
                counter += 1

        return counter / 100



qdarray = [0] * 500
qdarray1 = [0] * 500
qdarray2 = [0] * 500
qdarray3 = [0] * 500
qdarray4 = [0] * 500
edgearray = [0] * 500
circlearray = [0] * 500

for t in range(10):
    for s in range(500):
        edgearray[s] = edgearray[s] + (findedgeratio(np.random.uniform(0,1,[100,s+1]), (s+1)))/10

for e in range(10):
    for q in range(500):
     circlearray[q] = circlearray[q] + findcircleratio(np.random.uniform(0,1,[100,q+1]), (q+1))/10

for a in range(5):
    for b in range(500):
        qdarray[b] = qdarray[b] + findratio(np.random.uniform(0, 1, [10, b+1]), 10)/5
    for c in range(500):
        qdarray1[c] = qdarray[c] + findratio(np.random.uniform(0,1, [20, c+1]), 20)/5
    for d in range(500):
        qdarray2[d] = qdarray2[d] + findratio(np.random.uniform(0, 1, [30, d + 1]), 30)/5
    for e in range(500):
        qdarray3[e] = qdarray2[e] + findratio(np.random.uniform(0, 1, [40, e + 1]), 40)/5
    for f in range(500):
        qdarray4[f] = qdarray4[f] + findratio(np.random.uniform(0, 1, [50, f + 1]), 50)/5

    # Create a square figure
    q1a = plt.figure(figsize=(5, 5))
    q1b = plt.figure(figsize=(5,5))
    q2a = plt.figure(figsize=(5,5))
    q2b = plt.figure(figsize=(5,5))
    xarray = []
    for i in range(1, 501):
        xarray.append(i)

    some_xs = xarray



    ax_1a = q1a.add_subplot(1,1,1)
    ax_1a.set_xlabel("dimension")
    ax_1a.set_ylabel("expected ratio")
    ax_1a.scatter(some_xs, qdarray, s=1)


    ax_1b = q1b.add_subplot(1,1,1)
    ax_1b.set_xlabel("dimension")
    ax_1b.set_ylabel("expected ratio")
    ax_1b.scatter(some_xs, qdarray, s=1)
    ax_1b.scatter(some_xs, qdarray1, s=1)
    ax_1b.scatter(some_xs, qdarray2, s=1)
    ax_1b.scatter(some_xs, qdarray3, s=1)
    ax_1b.scatter(some_xs, qdarray4, s=1)


    ax_2a = q2a.add_subplot(1,1,1)
    ax_2a.set_xlabel("dimension")
    ax_2a.set_ylabel("expected ratio")
    ax_2a.scatter(some_xs, edgearray, s=1)


    ax_circle = q2b.add_subplot(1,1,1)
    ax_circle.set_xlabel("dimension")
    ax_circle.set_ylabel("expected ratio")
    ax_circle.scatter(some_xs, circlearray, s=1)


    # Arrange everything in the plot such that we minimize overlap issues:
    plt.tight_layout()
    # Save the plot to disk, too:
    plt.savefig("output.pdf")
    # Show the plot:
    plt.show()
