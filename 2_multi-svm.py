from sklearn import svm

# initialize data
X_train = []
Y_train = []
X_test = []
Y_test = []

with open("dermTrain.csv") as trainFile:
    for line in trainFile:
        x_vector = line.split(",")[:-1]
        for i in range(len(x_vector)):
            x_vector[i] = int(x_vector[i])
        y = int(line.split(",")[-2])

        X_train.append(x_vector)
        Y_train.append(y)

with open("dermTest.csv") as testFile:
    for line in testFile:
        x_vector = line.split(",")[:-1]
        for i in range(len(x_vector)):
            x_vector[i] = int(x_vector[i])
        y = int(line.split(",")[-2])

        X_test.append(x_vector)
        Y_test.append(y)

# train svm - one versus one
ovo_clf = svm.SVC(decision_function_shape='ovo')
ovo_clf.fit(X_train, Y_train)
print("One-versus-One model count:", ovo_clf.decision_function([X_test[1]]).shape[1])

# train svm - one versus rest
ovr_clf = svm.SVC(decision_function_shape='ovr')
ovr_clf.fit(X_train, Y_train)
print("One-versus-Rest model count:", ovr_clf.decision_function([X_test[1]]).shape[1])

# test support vector machines
ovo_total = 0
ovr_total = 0
for i in range(len(X_test)):
    if ovo_clf.predict([X_test[i]]) == Y_test[i]:
        ovo_total = ovo_total + 1
    if ovr_clf.predict([X_test[i]]) == Y_test[i]:
        ovr_total = ovr_total + 1

print("OVO performance: ", ovo_total / len(X_test), "%")
print("OVR performance:", ovr_total / len(X_test), "%")




