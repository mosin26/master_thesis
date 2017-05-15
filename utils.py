import time
import pycosat
import numpy as np


def decide_var(cnf, clf, var, trials=5, deep=1, to_sat=True):
    pos_probas, neg_probas = [], []
    to_sat = int(to_sat)

    pos_cnf = cnf.set_var(var)
    for j in range(trials):
        temp_cnf = pos_cnf
        if [] in temp_cnf.clauses:
            pos_proba = 0
        else:
            pos_proba = clf.predict_proba([temp_cnf.get_features()])[0][to_sat]
        for i in range(deep):
            temp_cnf = temp_cnf.set_var()
            if [] in temp_cnf.clauses:
                pos_proba *= 0
            else:
                pos_proba *= clf.predict_proba([temp_cnf.get_features()])[0][to_sat]
        pos_probas.append(pos_proba)

    neg_cnf = cnf.set_var(-var)
    for j in range(trials):
        temp_cnf = neg_cnf
        if [] in temp_cnf.clauses:
            neg_proba = 0
        else:
            neg_proba = clf.predict_proba([temp_cnf.get_features()])[0][to_sat]
        for i in range(deep):
            temp_cnf = temp_cnf.set_var()
            if [] in temp_cnf.clauses:
                neg_proba *= 0
            else:
                neg_proba *= clf.predict_proba([temp_cnf.get_features()])[0][to_sat]
        neg_probas.append(neg_proba)

    return int(np.sign(np.mean(pos_probas)-np.mean(neg_probas)))


def preprocessing(cnf, clf, trials=5, deep=1, to_sat=True):
    start_time = time.time()
    assignment = []
    for var in cnf.variables:
        assignment.append(var * decide_var(cnf, clf, var, trials, deep, to_sat))
    end_time = time.time()
    duration = end_time - start_time
    backbones = get_backbones(cnf)
    accuracy = float(len([var for var in assignment if var in backbones])) / len(backbones)
    return assignment, accuracy, duration


def get_backbones(cnf):
    solutions = list(pycosat.itersolve(cnf.clauses))
    all_vars = [set([solution[i] for solution in solutions]) for i in range(len(solutions[0]))]
    backbones = [i.pop() for i in all_vars if len(i) == 1]
    return backbones
