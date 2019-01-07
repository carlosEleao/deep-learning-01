
import os
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report


classes = {"Chevrolet_Onix": 0, "Ford_Ka": 1, "Hyundai_HB20": 2,
               "Renault_Sandero": 3, "Volkswagen_Gol": 4}

def check_is_npz(path):
    extensions = {".npz"}
    return any(path.endswith(ext) for ext in extensions)


def class_to_num(path):
    
    for car in classes:
        if car in path:
            return classes[car]


def find_paths(path):
    paths = []
    level_a = os.listdir(path)
    for level_name in level_a:
        if not os.path.isdir(os.path.join(path, level_name)):
            continue
        for features in os.listdir(os.path.join(path, level_name)):
            npz_path = os.path.join(path, level_name, features)
            if check_is_npz(npz_path):
                paths += [npz_path]
                
    return paths


def load_features(paths):
    all_features = []
    all_classes = []
    for path in paths:
        print path
        all_features += [np.load(path)["f1"][0]]
        all_classes += [class_to_num(path)]

    print all_features
    print all_classes
    return np.array(all_features), np.array(all_classes)


def main():
    base_path = "/Users/carlosleao/workspace/ml-deeplearning/datasets/DeepLearningFiles/"
    features_paths = find_paths(base_path)
    X, y = load_features(features_paths)

    
    # X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)   

    best_score = {"score": 0., "gamma": '', "kernel": '', "C": ''}
    kernels = ["linear", "rbf", "poly"]
    gammas = [0.1, 1, 10, 100]
    cs = [0.1, 1, 10, 100, 1000]

    # for kernel in kernels:
    #     for gamma in gammas:
    #         for c in cs:
    #             clf = SVC(gamma=gamma, kernel=kernel, C=c)
    #             clf.fit(X_train, y_train)
    #             scores = cross_val_score(clf, X, y, cv=5)
    #             # for feature in X[0]:
    #             #     print feature
    #             print classification_report(y_test, clf.predict(X_test), target_names=classes.items())
    #             score = clf.score(X_test,y_test)
    #             if best_score["score"] < score:
    #                 best_score["score"] = score
    #                 best_score["gamma"] = gamma
    #                 best_score["kernel"] = kernel
    #                 best_score["C"] = c

    clf = SVC(gamma=0.1, kernel='rbf', C=100)
    clf.fit(X_train, y_train)
    
    best_score = cross_val_score(clf, X, y, cv=5)

    print classification_report(y_test, clf.predict(X_test), target_names=classes.items())
    print("Best Score:",best_score)


if __name__ == "__main__":
    main()
