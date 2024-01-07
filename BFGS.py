import pennylane.numpy as np 
from pennylane.optimize import GradientDescentOptimizer


class BFGSOptimizer(GradientDescentOptimizer):
    
    def __init__(self, old_hessian = None, old_grad = None):
        """Initializes BFGS optimizer"""
        self.old_hessian = old_hessian
        self.old_grad = old_grad
        self.a = 1
        self.c1 = 1e-4 
        self.c2 = 0.9 
        self.stepsize = None

    def step(self, objective_fn, *args, grad_fn=None, **kwargs):
        """
        Args: - objective_fn to be minimized
              - theta
        Returns:
              - updated theta
              
        """
        if type(self.old_hessian) == type(None):
            self.old_hessian = np.eye(len(args[0].flatten()))
        if type(self.old_grad) == type(None):
            g, _ = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
            self.old_grad = g[0].flatten()
        
        new_args = self.bfgs_update(objective_fn, args, grad_fn = grad_fn, **kwargs)
        
        return new_args
    
    def step_and_cost(self, objective_fn, *args, grad_fn=None, **kwargs):
        """
        Args: - objective_fn to be minimized
              - variables
              - grad_fn if available
              - other parameters needed for the function
        Returns:
              - updated theta, cost of updated theta
        """
        if type(self.old_hessian) == type(None):
            self.old_hessian = np.eye(len(args[0].flatten()))
        if type(self.old_grad) == type(None):
            g, _ = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
            self.old_grad = g[0].flatten()
        
        new_args = self.bfgs_update(objective_fn, args, grad_fn = grad_fn, **kwargs)
        
        forward = objective_fn(*args, **kwargs)
        
        return new_args, forward   

    def line_search(self, objective_fn, *args, search_direction, grad_fn=None, **kwargs):
        '''
        BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
        '''
        a = self.a
        x, temp, sm_param = args[0]
        search_direction = np.reshape(search_direction, np.shape(x))
        fx = objective_fn(x, temp, sm_param, **kwargs)
        x_new = x + self.a * search_direction
        new_args = x_new, temp, sm_param
        g_new, _ = self.compute_grad(objective_fn, new_args, kwargs, grad_fn=grad_fn) ### just removed kwargs
        g_new = g_new[0].flatten()
        
        num_it = 0
        
        while objective_fn(x_new, temp, sm_param, **kwargs) >= fx + (self.c1 * a * np.dot(np.transpose(self.old_grad), search_direction.flatten())) or np.transpose(g_new) @ search_direction.flatten() <= self.c2 * np.dot(np.transpose(self.old_grad), search_direction.flatten()):
            num_it += 1
            a *= 0.5
            if num_it == 10:
                break
            x_new = x + a * search_direction
            new_args = x_new, temp, sm_param
            g_new, _ = self.compute_grad(objective_fn, new_args, kwargs, grad_fn=grad_fn)
            g_new = g_new[0].flatten()
            
        return a
    
    def bfgs_update(self, objective_fn, *args, grad_fn=None, **kwargs):
        '''
        Do BFGS update ---- See Nocedal 
        '''        
        search_direction = -np.dot(self.old_hessian, self.old_grad)
        a = self.line_search(objective_fn, *args, search_direction = search_direction, grad_fn = grad_fn, **kwargs)
        s = a * search_direction
        x, temp, sm_param = args[0]
        search_direction = np.reshape(search_direction, np.shape(x))
        x_new = x + a * search_direction
        
        new_args = x_new, temp, sm_param
        g_new, _ = self.compute_grad(objective_fn, new_args, kwargs, grad_fn=grad_fn)
        g_new = g_new[0].flatten()
        y = g_new - self.old_grad 
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(len(args[0][0].flatten()),1))
        s = np.reshape(s,(len(args[0][0].flatten()),1))
        r = 1/(np.dot(np.transpose(y), s))
        li = (np.eye(len(args[0][0].flatten()))-(r*((np.dot(s, np.transpose(y))))))
        ri = (np.eye(len(args[0][0].flatten()))-(r*((np.dot(y, np.transpose(s))))))
        
        
        hess_inter = np.dot(li, 
                            np.dot(self.old_hessian, ri)
                            )
        self.previous_hessian = self.old_hessian
        self.previous_gradient = self.old_grad
        self.old_hessian = hess_inter + (r*(np.dot(s, np.transpose(s)))) # BFGS Update
        self.old_grad = g_new[:]
        self.stepsize = a
        
        new_args = x_new[:]
        
        return new_args







