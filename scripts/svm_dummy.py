
from sklearn import svm
from sklearn.model_selection import train_test_split

X = # Feature vectors of reactions
Y = # Vector of Target Labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def my_kernel(X, Y):
    """
    We create a custom kernel
    """

    return np.dot(X, Y)


# we create an instance of SVM and fit out data.
clf = svm.SVC(kernel=my_kernel)

clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)


