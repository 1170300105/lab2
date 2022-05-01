import cvxpy as cp
import numpy as np
import time


# f(x) = 1.5 * x1 ** 2 + x2 ** 2 + 0.5 * x3 ** 2 + x1 + x2 + x3
# s.t. x1 + x2 + x3 = 1
class AlmProblem:
    x1 = cp.Variable()
    x2 = cp.Variable()
    x3 = cp.Variable()

    def __init__(self, v, c):
        self.v = v
        self.c = c

    def get_obj_x(self):
        x1 = self.x1
        x2 = self.x2
        x3 = self.x3
        self.x3.value = 0
        self.x2.value = 0
        self.x1.value = 0
        return cp.Minimize(1.5 * x1 ** 2 + x2 ** 2 + 0.5 * x3 ** 2 + x1 + x2 + x3
                           + self.v * (x1 + 2 * x2 + x3 - 4)
                           + (x1 + 2 * x2 + x3 - 4) ** 2 / (2 * self.c))

    def get_obj_u(self):
        x1 = self.x1.value
        x2 = self.x2.value
        x3 = self.x3.value
        return cp.Minimize(1.5 * x1 ** 2 + x2 ** 2 + 0.5 * x3 ** 2 - x1 * x2 - x2 * x3 + x1 + x2 + x3
                           + self.u * (x1 + 2 * x2 + x3 - 4)
                           + self.c * (x1 + 2 * x2 + x3 - 4) ** 2 / 2)

    def get_x(self):
        return [self.x1.value, self.x2.value, self.x3.value]

    def step(self):
        problem = cp.Problem(self.get_obj_x())
        s = problem.solve()
        x1 = self.x1.value
        x2 = self.x2.value
        x3 = self.x3.value
        self.v += self.c * (x1 + 2 * x2 + x3 - 4)
        return s


def alm_method():
    searchpath = []
    t = time.time()
    almp = AlmProblem(0.1, 1)
    for i in range(20):
        almp.step()
        x1 = almp.x1.value
        x2 = almp.x2.value
        x3 = almp.x3.value
        v = 1.5 * x1 ** 2 + x2 ** 2 + 0.5 * x3 ** 2 + x1 + x2 + x3
        h = x1 + 2 * x2 + x3 - 4
        searchpath.append([x1, x2, x3])
        if abs(h) < 1e-6:
            break
    searchpath.append(time.time() - t)
    return searchpath


# f(x) = 1.5 * x1 ** 2 + x2 ** 2  + x1 + x2 +  0.5 * x3 ** 2 + x3
# s.t. x1 + x2 + x3 = 1
class AdmmProblem:
    x1 = cp.Variable()
    x2 = cp.Variable()
    x3 = cp.Variable()

    def __init__(self, v, c):
        self.v = v
        self.c = c
        self.x3.value = 0
        self.x2.value = 0
        self.x1.value = 0

    def get_obj_x(self):
        x1 = self.x1
        x2 = self.x2
        x3 = self.x3.value
        return cp.Minimize(1.5 * x1 ** 2 + x2 ** 2 + 0.5 * x3 ** 2 + x1 + x2 + x3
                           + self.v * (x1 + 2 * x2 + x3 - 4)
                           + (x1 + 2 * x2 + x3 - 4) ** 2 / (2 * self.c))

    def get_obj_z(self):
        x1 = self.x1.value
        x2 = self.x2.value
        x3 = self.x3
        return cp.Minimize(1.5 * x1 ** 2 + x2 ** 2 + 0.5 * x3 ** 2  + x1 + x2 + x3
                           + self.v * (x1 + 2 * x2 + x3 - 4)
                           + (x1 + 2 * x2 + x3 - 4) ** 2 / (2 * self.c))

    def get_x(self):
        return [self.x1.value, self.x2.value, self.x3.value]

    def step(self):
        problem = cp.Problem(self.get_obj_x())
        problem.solve()
        problem = cp.Problem(self.get_obj_z())
        problem.solve()
        x1 = self.x1.value
        x2 = self.x2.value
        x3 = self.x3.value
        self.v += self.c * (x1 + 2 * x2 + x3 - 4)



def admm_method():
    searchpath = []
    t = time.time()
    almp = AdmmProblem(0.1, 1)
    for i in range(20):
        almp.step()
        x1 = almp.x1.value
        x2 = almp.x2.value
        x3 = almp.x3.value
        v = 1.5 * x1 ** 2 + x2 ** 2 + 0.5 * x3 ** 2 + x1 + x2 + x3
        h = x1 + 2 * x2 + x3 - 4
        searchpath.append([x1, x2, x3])
        if abs(h) < 1e-6:
            break
    searchpath.append(time.time()-t)
    return searchpath


if __name__ == '__main__':
    path = alm_method()
    for p in path:
        print(p)
    path = admm_method()
    for p in path:
        print(p)
