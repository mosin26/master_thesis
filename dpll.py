import time
import operator
import numpy as np
from random import choice
from utils import decide_var


class DPLL:
    def __init__(self, cnf, branching, use_classifier=False, clf=None, trials=5, deep=1, to_sat=True):
        self.cnf = cnf
        self.branching = branching
        self.use_classifier = use_classifier
        self.trials = trials
        self.clf = clf
        self.deep = deep
        self.to_sat = to_sat
        self.conflicts = None
        self.result = None
        self.computation_time = None

    @staticmethod
    def unit_propagate(cnf):
        had_unit_clause = True
        while [] not in cnf.clauses and had_unit_clause:
            had_unit_clause = False
            clauses_length = np.array([len(clause) for clause in cnf.clauses])
            unit_ind = np.argwhere(clauses_length == 1)
            if len(unit_ind) > 0:
                had_unit_clause = True
                cnf = cnf.set_var(cnf.clauses[unit_ind[0][0]][0])
        return cnf

    def dpll(self, cnf):
        cnf = self.unit_propagate(cnf)
        if cnf.m == 0:
            return 1
        if [] in cnf.clauses:
            self.conflicts += 1
            return 0

        if self.branching == 'random':
            x = choice(cnf.variables)
            if self.use_classifier:
                x *= decide_var(self.cnf, self.clf, x, self.trials, self.deep, self.to_sat)

        elif self.branching == 'maxo':
            occurrences = {i: 0 for i in cnf.variables}
            for clause in cnf.clauses:
                for var in clause:
                    occurrences[abs(var)] += 1
            x = sorted(occurrences.items(), key=operator.itemgetter(1))[-1][0]
            if self.use_classifier:
                x *= decide_var(self.cnf, self.clf, x, self.trials, self.deep, self.to_sat)

        elif self.branching == 'moms':
            occurrences = {i: 0 for i in cnf.variables}
            min_clause = min([len(clause) for clause in cnf.clauses])
            for clause in cnf.clauses:
                if len(clause) == min_clause:
                    for var in clause:
                        occurrences[abs(var)] += 1
            x = sorted(occurrences.items(), key=operator.itemgetter(1))[-1][0]
            if self.use_classifier:
                x *= decide_var(self.cnf, self.clf, x, self.trials, self.deep, self.to_sat)

        else:
            print 'incorrect branching rule.. exit'
            return

        if self.dpll(cnf.set_var(x)) == 1:
            return 1
        if self.dpll(cnf.set_var(-x)) == 1:
            return 1
        return 0

    def solve(self):
        self.conflicts = 0
        start_time = time.time()
        self.result = self.dpll(self.cnf)
        end_time = time.time()
        self.computation_time = end_time - start_time
