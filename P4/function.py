import numpy as np

class QuadraticFunction(object):
    def __init__(self, H, g, A, b, C=None, d=None):
        self.H = H
        self.g = g
        self.A = A
        self.b = b

    def __call__(self, x):
        return 0.5 * x.T @ self.H @ x + self.g.T @ x

    def gradient(self, x):
        return self.H @ x + self.g

    def hessian(self):
        return self.H

    def constraints_eq(self, x):
        return self.A @ x - self.b
    
    def constraints_ineq(self, x):
        return self.C @ x - self.d if self.C is not None else None
    
    def jacobian(self, x):
        return self.A
    
if __name__ == '__main__':
    H = np.array([[2, 0], [0, 2]])
    g = np.array([-2, -5])
    A = np.array([[1, 1], [1, -1]])
    b = np.array([1, 0])
    
    f = function(H, g, A, b)
    
    x = np.array([0.5, 0.5])
    
    print("Function value:", f(x))
    print("Gradient:", f.gradient(x))
    print("Hessian:", f.hessian())
    print("Constraints:", f.constraints(x))