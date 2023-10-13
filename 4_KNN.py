from statistics import mode
import pandas

class Global:
    view1data = []
    view2data = []
    view3data = []
    view4data = []
    viewSet = [view1data, view2data, view3data, view4data]
    solutionVector = []
    view1cols = [1, 2, 3, 4, 5]  # clinical view
    view3cols = [6, 7, 8, 9, 10, 11]  # clinical view
    view2cols = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # histopathological view
    view4cols = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 ,34]


def reset():
    Global.view1cols = []
    Global.view2cols = []
    Global.view3cols = []
    Global.view4cols = []

    Global.view1data = []
    Global.view2data = []
    Global.view3data = []
    Global.view4data = []
    Global.solutionVector = []

def init():
    with open("dermatology.txt", "r") as data:
        i = 0
        for line in data:
            i = i + 1
            row = line.split(",")
            vector1 = []
            vector2 = []
            vector3 = []
            vector4 = []

            for i in Global.view1cols:
                vector1.append(row[i - 1])
            for i in Global.view2cols:
                vector2.append(row[i - 1])
            for i in Global.view3cols:
                vector3.append(row[i - 1])
            for i in Global.view4cols:
                if i == 34:
                    vector4.append(3/85 * int(row[i - 1]))
                else:
                    vector4.append(row[i - 1])

            Global.view1data.append(vector1)
            Global.view2data.append(vector2)
            Global.view3data.append(vector3)
            Global.view4data.append(vector4)
            Global.solutionVector.append(int(row[-1]))


def l2norm(v1, v2):  # l1 norm
    total = 0
    for m in range(len(v1)):
        total = total + (int(v1[m]) - int(v2[m])) ** 2
    return total


def vote(vector):
    return mode(vector)


def predict(input, a, b):  # input is an integer demarcating a row in the table
    viewVotes = []
    view_no = 0
    for view in Global.viewSet:
        view_no = view_no + 1
        distances = []
        numbers = (num for num in range(len(view)) if num != input)
        for j in numbers:
            distances.append((Global.solutionVector[j], l2norm(view[j], view[input])))
        distances.sort(key=lambda x: x[1])  # sort in ascending order, based on a distance value
        neighbors = []
        if view_no == 1:
            k = a
        else:
            k = b
        for i in range(k):  # get first 5 samples
            neighbors.append(int(distances[i][0]))
        viewVotes.append(vote(neighbors))
    return vote(viewVotes)


def tune():
    max_acc = 0
    max_a = 0
    max_b = 0
    for a in range(1, 9):
        for b in range(1, 9):
            total = 0
            for i in range(len(Global.solutionVector)):
                if predict(i, a, b) == Global.solutionVector[i]:
                    total = total + 1

            if total > max_acc:
                max_acc = total
                max_a = a
                max_b = b

            print("|", end="")
        print(a)
    print(max_acc / len(Global.solutionVector), max_a, max_b)


def test():
    # 94.3% accuracy
    # parameters for predict can be obtained from tune()
    total = 0
    for i in range(0, 300):
        if predict(i, 6, 3) == Global.solutionVector[i]:
            total = total + 1

    print(total / 300)
    return (total / 300)


if __name__ == "__main__":
   init()
   test()
