import numpy as np

class SQPHimmelblau:    
    def __init__(self):
        pass
    
    def __call__(self, x):
        """
        Himmelblau's function.
        """
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    
    def hessian(self, x):
        """
        Hessian of Himmelblau's function.
        """
        x1, x2 = x
        h11 = 12*x1**2 + 4*x2 - 42
        h12 = 4*x1 + 4*x2
        h22 = 4*x1 + 12*x2**2 - 26
        return np.array([[h11, h12], [h12, h22]])
    
    def gradient(self, x):
        """
        Gradient of Himmelblau's function.
        """
        x1, x2 = x
        df_dx1 = 4*x1*(x1**2 + x2 - 11) + 2*(x1 + x2**2 - 7)
        df_dx2 = 2*(x1**2 + x2 - 11) + 4*x2*(x1 + x2**2 - 7)
        return np.array([df_dx1, df_dx2])
    
    def h(self, x):
        x1, x2 = x
        return np.array([x1 - 3/2 * x2])
    
    def h_diff(self, x):
        x1, x2 = x
        return np.array([[1.0, -3/2]])
    
    def g1(self, x):
        x1, x2 = x
        return np.array([(x1 + 2)**2 - x2])
    
    def g1_diff(self, x):
        x1, x2 = x
        return np.array([2*(x1 + 2), -1.0])
    
    def g2(self, x):
        x1, x2 = x
        return np.array([-4*x1 + 10*x2])
    
    def g2_diff(self, x):
        x1, x2 = x
        return np.array([-4.0, 10.0])
    
    def lagrangian(self, x, lam):
        """
        Lagrangian function.
        """
        return self(x) + lam[0] * self.h(x) + lam[1] * self.g1(x) + lam[2] * self.g2(x)
    
    def lagrangian_gradient(self, x, lam):
        """
        Gradient of the Lagrangian function.
        """
        grad = self.gradient(x)
        grad += lam[0] * self.h_diff(x).flatten()
        grad += lam[1] * self.g1_diff(x).flatten()
        grad += lam[2] * self.g2_diff(x).flatten()
        return grad
    
    def lagrangian_hessian(self, x, lam):
        """
        Hessian of the Lagrangian function.
        """
        hess = self.hessian(x)
        hess = hess.astype(np.float64)
        hess += lam[1] * np.array([[2, 0], [0, 0]])
        return hess
    
class SQPHimmelblau_lagrangian:
    def __init__(self, lambdas):
        self.lam = lambdas
    
    def __call__(self, x):
        """
        Lagrangian function.
        """
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2 + self.lam[0] * (x[0] - 3/2 * x[1]) + self.lam[1] * ((x[0] + 2)**2 - x[1]) + self.lam[2] * (-4*x[0] + 10*x[1])
    
    def gradient(self, x):
        """
        Gradient of the Lagrangian function.
        """
        x1, x2 = x
        df_dx1 = 4*x1*(x1**2 + x2 - 11) + 2*(x1 + x2**2 - 7) + self.lam[0] * 1.0 + self.lam[1] * 2*(x1 + 2) - self.lam[2] * 4.0
        df_dx2 = 2*(x1**2 + x2 - 11) + 4*x2*(x1 + x2**2 - 7) - self.lam[0] * (3/2) - self.lam[1] * 1.0 + self.lam[2] * 10.0
        return np.array([df_dx1, df_dx2])

class SQPBeale:
    def __init__(self):
        pass
    
    def __call__(self, x):
        """
        Beale's function.
        """
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*(x[1]**2))**2 + (2.625 - x[0] + x[0]*(x[1]**3))**2
    
    def hessian(self, x):
        """
        Hessian of Beale's function.
        """
        x1, x2 = x
        h11 = 2*(x2 - 1)**2 + 2*(x2**2 - 1)**2 + 2*(x2**3 - 1)**2
        h12 = 2*x1*(-1 + x2) + 3.0 - 2*x1 + 2*x1*x2 + 4*x1*x2*(x2**2 - 1) + 4*(2.25 - x1 + x1*x2**2)*x2 + 6*x1*x2**2*(x2**3 - 1) + 6*(2.625 - x1 + x1*x2**3)*x2**2
        h22 = 2*x1**2 + 8*x1**2*x2**2 + 4*(2.25 - x1 + x1*x2**2)*x1 + 18*x1**2*x2**4 + 12*(2.625 - x1 + x1*x2**3)*x1*x2

        return np.array([[h11, h12], [h12, h22]])
    
    def gradient(self, x):
        """
        Gradient of Beale's function.
        """
        x1, x2 = x
        df_dx1 = 2*(1.5 - x1 + x1*x2) * (-1 + x2) + 2*(2.25 - x1 + x1*(x2**2)) * (-1 + x2**2) + 2*(2.625 - x1 + x1*(x2**3)) * (-1 + x2**3)
        df_dx2 = 2*(1.5-x1+x1*x2) * x1 + 4*(2.25-x1+x1*(x2**2)) * x1*x2 + 6*(2.625-x1+x1*(x2**3)) * x1*x2**2
        return np.array([df_dx1, df_dx2])
    
    def h(self, x):
        return None
    
    def h_diff(self, x):
        return None
    
    def g(self, x):
        x1, x2 = x
        return np.array([-x1**2 - x2**2 + 12])
    
    def g_diff(self, x):
        x1, x2 = x
        return np.array([-2*x1, -2*x2])
    
    def lagrangian(self, x, lam):
        """
        Lagrangian function.
        """
        return self(x) + lam * self.g(x)
    
    def lagrangian_hessian(self, x, lam):
        """
        Hessian of the Lagrangian function.
        """
        hess = self.hessian(x)
        hess = hess.astype(np.float64)
        hess += lam * np.array([[2, 0], [0, 2]])
        return hess
    
    def lagrangian_gradient(self, x, lam):
        """
        Gradient of the Lagrangian function.
        """
        grad = self.gradient(x)
        grad += lam * self.g_diff(x).flatten()
        return grad
    
class SQPBeale_lagrangian:
    def __init__(self, lambdas):
        self.lam = lambdas
    
    def __call__(self, x):
        """
        Lagrangian function.
        """
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*(x[1]**2))**2 + (2.625 - x[0] + x[0]*(x[1]**3))**2 + self.lam * (-x[0]**2 -x[1]**2 + 12)
    
    def gradient(self, x):
        """
        Gradient of the Lagrangian function.
        """
        x1, x2 = x
        df_dx1 = 2*(1.5 - x1 + x1*x2) * (-1 + x2) + 2*(2.25 - x1 + x1*(x2**2)) * (-1 + x2**2) + 2*(2.625 - x1 + x1*(x2**3)) * (-1 + x2**3) - 2*self.lam*x1
        df_dx2 = 2*(1.5-x1+x1*x2) * x1 + 4*(2.25-x1+x1*(x2**2)) * x1*x2 + 6*(2.625-x1+x1*(x2**3)) * x1*x2**2 - 2*self.lam*x2
        return np.array([df_dx1, df_dx2])
        