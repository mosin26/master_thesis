import random
import numpy as np
from scipy.stats import entropy


class CNF:
    def __init__(self, path=None, clauses=None):
        if path:
            with open(path, 'r') as cnf:
                formula = cnf.read()
                formula = formula.split('\n')
                start_index = 0
                while formula[start_index][0] != 'p':
                    start_index += 1
                self.n = int(formula[start_index].split()[2])
                self.variables = [i+1 for i in range(self.n)]
                self.m = int(formula[start_index].split()[3])
                self.clauses = [list(map(int, formula[start_index + 1 + i].split()[:-1])) for i in range(self.m)]
        else:
            variables = set()
            for clause in clauses:
                for var in clause:
                    variables.add(abs(var))
            self.n = len(variables)
            self.variables = list(variables)
            self.m = len(clauses)
            self.clauses = clauses

    def get_size(self):
        return [self.m, self.n, float(self.m) / self.n]

    def get_vc(self):
        nodes = {i: set() for i in self.variables}
        for j in range(self.m):
            for var in self.clauses[j]:
                nodes[abs(var)].add(j)
        nodes = [len(nodes.get(i)) for i in nodes]
        nodes_np = np.array(nodes)
        nodes_proba = np.unique(nodes_np, return_counts=True)[1]/float(len(nodes_np))
        nodes_entropy = entropy(list(nodes_proba))
        clause = []
        for j in range(self.m):
            cl = set()
            for i in range(len(self.clauses[j])):
                cl.add(abs(self.clauses[j][i]))
            clause.append(len(cl))
        clause_np = np.array(clause)
        clause_proba = np.unique(clause_np, return_counts=True)[1]/float(len(clause_np))
        clause_entropy = entropy(list(clause_proba))
        return [nodes_np.mean(), nodes_np.std()/nodes_np.mean(), nodes_np.min(), nodes_np.max(), nodes_entropy,
                clause_np.mean(), clause_np.std()/clause_np.mean(), clause_np.min(), clause_np.max(), clause_entropy]

    def get_v(self):
        variables = {i: set() for i in self.variables}
        for j in range(self.m):
            for var in self.clauses[j]:
                for var_o in self.clauses[j]:
                    if abs(var_o) != abs(var):
                        variables[abs(var)].add(abs(var_o))
        var_deg = [len(variables.get(i)) for i in variables]
        var_deg_np = np.array(var_deg)
        return [var_deg_np.mean(), var_deg_np.std()/var_deg_np.mean(), var_deg_np.min(), var_deg_np.max()]

    def get_balance(self):
        ratio_clause = []
        for clause in self.clauses:
            pos, neg = 0, 0
            for var in clause:
                if var > 0:
                    pos += 1
                else:
                    neg += 1
            ratio_clause.append(float(pos) / (pos + neg))
        ratio_clause_np = np.array(ratio_clause)
        ratio_clause_proba = np.unique(ratio_clause_np, return_counts=True)[1] / float(len(ratio_clause_np))
        ratio_clause_entropy = entropy(list(ratio_clause_proba))
        ration_var = {i: [0, 0] for i in self.variables}
        for j in range(self.m):
            for var in self.clauses[j]:
                if var > 0:
                    ration_var.get(abs(var))[0] += 1
                else:
                    ration_var.get(abs(var))[1] += 1
        ration_var = [float(ration_var.get(i)[0]) / (ration_var.get(i)[0] + ration_var.get(i)[1]) for i in ration_var]
        ration_var_np = np.array(ration_var)
        ration_var_proba = np.unique(ration_var_np, return_counts=True)[1] / float(len(ration_var_np))
        ration_var_entropy = entropy(list(ration_var_proba))
        binary, ternary = 0, 0
        for clause in self.clauses:
            if len(clause) == 2:
                binary += 1
            elif len(clause) == 3:
                ternary += 1
        return [ratio_clause_np.mean(), ratio_clause_np.std()/ratio_clause_np.mean(), ratio_clause_entropy,
                ration_var_np.mean(), ration_var_np.std()/ration_var_np.mean(), ration_var_np.min(),
                ration_var_np.max(), ration_var_entropy, float(binary)/self.m, float(ternary)/self.m]

    def get_horn(self):
        num_of_horns = 0
        horn_var = {i: 0 for i in self.variables}
        for clause in self.clauses:
            horn = True
            cnt = 0
            for var in clause:
                if var > 0:
                    cnt += 1
                if cnt > 1:
                    horn = False
                    break
            if horn:
                num_of_horns += 1
                for vr in clause:
                    horn_var[abs(vr)] += 1
        horn_var = [horn_var.get(i) for i in horn_var]
        horn_var_np = np.array(horn_var)
        horn_var_proba = np.unique(horn_var_np, return_counts=True)[1] / float(len(horn_var_np))
        horn_var_entropy = entropy(list(horn_var_proba))
        return [float(num_of_horns) / self.m, horn_var_np.mean(), horn_var_np.std()/horn_var_np.mean(),
                horn_var_np.min(), horn_var_np.max(), horn_var_entropy]

    def get_features(self):
        size = self.get_size()
        vc = self.get_vc()
        v = self.get_v()
        balance = self.get_balance()
        horn = self.get_horn()
        return size + vc + v + balance + horn

    def set_var(self, var=None):
        if not var:
            var = random.choice(self.variables + [-i for i in self.variables])
        new_clauses = [[i for i in clause if i != -var] for clause in self.clauses if var not in clause]
        return CNF(clauses=new_clauses)
