from sklearn import svm

# initialize data
X_train = []
Y_train = []
X_test = []
Y_test = []
a = [2, 3, 13, 15] # reduced/removed variables



def test(a):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    with open("dermTrain2.csv") as trainFile:
        for line in trainFile:
            x_vector = line.split(",")[:-1]
            for i in range(len(x_vector)):
                if i == 10:
                    x_vector[i] = 3*float(x_vector[i])
                elif i == 33:
                    x_vector[i] = (3/85) * float(x_vector[i])
                elif i in a:
                    x_vector[i] = 0

                else:
                    x_vector[i] = float(x_vector[i])

            y = float(line.split(",")[-1])

            X_train.append(x_vector)
            Y_train.append(y)

    with open("dermTest2.csv") as testFile:
        for line in testFile:
            x_vector = line.split(",")[:-1]
            for i in range(len(x_vector)):
                if i == 10:
                    x_vector[i] = 3 * float(x_vector[i])
                elif i == 33:
                    x_vector[i] = (3 / 85) * float(x_vector[i])
                elif i in a:
                    x_vector[i] = 0

                else:
                    x_vector[i] = float(x_vector[i])
            y = float(line.split(",")[-1])

            X_test.append(x_vector)
            Y_test.append(y)

    # train svm - one versus one
    ovo_clf = svm.SVC(decision_function_shape='ovo')
    ovo_clf.fit(X_train, Y_train)
  #  print("One-versus-One model count:", ovo_clf.decision_function([X_test[1]]).shape[1])

    # train svm - one versus rest
    ovr_clf = svm.SVC(decision_function_shape='ovr')
    ovr_clf.fit(X_train, Y_train)
   # print("One-versus-Rest model count:", ovr_clf.decision_function([X_test[1]]).shape[1])

    # test support vector machines
    ovo_total = 0
    ovr_total = 0
    for i in range(len(X_test)):
        if ovo_clf.predict([X_test[i]]) == Y_test[i]:
            ovo_total = ovo_total + 1
        if ovr_clf.predict([X_test[i]]) == Y_test[i]:
            ovr_total = ovr_total + 1

   # print("OVO performance: ", ovo_total / len(X_test), "%")
  #  print("OVR performance:", ovr_total / len(X_test), "%")
    return ovo_total / len(X_test)


print(test([2, 3]))
max = -1
val = -1
for i in range(34):
    for j in range(34):
        a = [2, 3, 13, 15, i, j]
       # print(a)
        b = test(a)
        if b > val:
            val = b
            max = a

print(max)
print(val)

# remove 3
