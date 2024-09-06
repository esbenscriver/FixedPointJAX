import jax.numpy as jnp
from jax import lax

def UpdateFixedPointRoot(inputTuple: tuple, fun: callable) -> tuple:
        """Update fixed-point

            - Inputs:
                - fun: function to iterate on
                - inputTuple: tuple
                    - x0: current value of x
                    - step_norm: norm of step
                    - root_norm: norm of roots
                    - step_tol: stopping tolerance for step
                    - root_tol: stopping tolerance for roots
                    - max_iter: maximum number of iterations
                    - i: number of succeded iterations

            - Output (tuple)
                - x1: updated value of x
                - step_norm: norm of step
                - root_norm: norm of roots
                - step_tol: stopping tolerance for step
                - root_tol: stopping tolerance for roots
                - max_iter: maximum number of iterations
                - i: number of succeded iterations
        """
        # Unpack input tuple
        (x0, step_norm, root_norm, step_tol, root_tol, max_iter, i, step_factor, step_min, step_max) = inputTuple
        x1, z = fun(x0) #update guess and root for the solution of the fixed-point equation

        # Evaluate norm of x and z
        step_norm = jnp.linalg.norm(x1 - x0) #stopping tolerance for step
        root_norm = jnp.linalg.norm(z) #stopping tolerance for roots

        # Pack tuple
        return (x1.copy(), step_norm, root_norm, step_tol, root_tol, max_iter, i + 1, step_factor, step_min, step_max)

def ConditionFixedPointRoot(out: tuple) -> bool:
    """ Conditions for continuation of while loop

            - Input:
                - x0: current value of x
                - step_norm: norm of step
                - root_norm: norm of roots
                - step_tol: stopping tolerance for step
                - root_tol: stopping tolerance for roots
                - max_iter: maximum number of iterations
                - i: number of succeded iterations

            - Output: boolean (true/false)

    """
    #Unpack input tuple
    (x0, step_norm, root_norm, step_tol, root_tol, max_iter, i, step_factor, step_min, step_max) = out

    # Evaluate stopping criterions
    cond1 = (root_norm > root_tol) #stop if root is zero
    cond2 = (step_norm > step_tol) #stop if step is zero
    cond3 = (i < max_iter) #stop if maximum iteration is reached
    cond4 = (jnp.any(jnp.isnan(x0)) == 0) #stop if any value is NaN

    cond_tol = jnp.logical_and(cond1, cond2) #step or root is close to zero
    return jnp.logical_and(jnp.logical_and(cond_tol, cond3), cond4)

def UpdateSQUAREM(inputTuple: tuple, fun: callable) -> tuple:
        """Update fixed-point

            - Inputs:
                - fun: function to iterate on
                - inputTuple: tuple
                    - x0 (array): current value of x
                    - step_norm (float): norm of step
                    - root_norm (float): norm of roots
                    - step_tol (float): stopping tolerance for step
                    - root_tol (float): stopping tolerance for roots
                    - max_iter (float): maximum number of iterations
                    - i (int): number of succeded iterations

            - Output (tuple)
                - x1 (array): updated value of x
                - step_norm (float): norm of step
                - root_norm (float): norm of roots
                - step_tol (float): stopping tolerance for step
                - root_tol (float): stopping tolerance for roots
                - max_iter (float): maximum number of iterations
                - i (int): number of succeded iterations
        """
        # Unpack input tuple
        (x0, step_norm, root_norm, step_tol, root_tol, max_iter, i, step_factor, step_min, step_max) = inputTuple
        
        x1 = fun(x0)[0] # first fixed-point step
        x2 = fun(x1)[0] # second fixed-point step

        # Accelerated step
        r = x1 - x0 # change
        v = x2 - x1 - r # curvature

        alpha =-jnp.sqrt(jnp.sum(r**2) / jnp.sum(v**2))

        x, z = fun(x0 - 2 * alpha * r + (alpha**2) * v)

        # Evaluate norm of step and root
        step_norm = jnp.linalg.norm(x - x0) # norm of step
        root_norm = jnp.linalg.norm(z) # norm of root

        # Pack tuple
        return (x.copy(), step_norm, root_norm, step_tol, root_tol, max_iter, i + 1, step_factor, step_min, step_max)

def ConditionSQUAREM(out: tuple) -> bool:
    """ Conditions for continuation of while loop

            - Input:
                - x0: current value of x
                - step_norm: norm of step
                - root_norm: norm of roots
                - step_tol: stopping tolerance for step
                - root_tol: stopping tolerance for roots
                - max_iter: maximum number of iterations
                - i: number of succeded iterations

            - Output: boolean (true/false)

    """
    #Unpack input tuple
    (x0, step_norm, root_norm, step_tol, root_tol, max_iter, i, step_factor, step_min, step_max)= out

    # Evaluate stopping criterions
    cond1 = (root_norm > root_tol) #stop if root norm is close to zero
    cond2 = (step_norm > step_tol) #stop if step norm is close to zero
    cond3 = (i < max_iter) #stop if maximum iteration is reached
    cond4 = (jnp.any(jnp.isnan(x0)) == 0) #stop if any value is NaN

    cond_tol = jnp.logical_and(cond1, cond2) #step norm or root norm is close to zero
    return jnp.logical_and(jnp.logical_and(cond_tol, cond3), cond4)

def FixedPointRoot(fun: callable,
                   x0: jnp.ndarray,
                   acceleration: str = "None",
                   step_tol: float = 1e-8,
                   root_tol: float = 1e-6,
                   max_iter: int = 1000) -> tuple:
    """Solve fixed-point equation by fixed point iterations

        - Inputs:
            fun: fixed-point equation of the form: x, z = fun(x)
            x0: initial guess
            step_tol: stopping tolerance for steps, x_{i+1} - x_{i}
            root_tol: stopping tolerance for roots, z_{i}
            max_iter: maximum number of iterations

        - Outputs:
            - x1: solution to fixed point equation
            - step_norm: norm of steps, x_{i+1} - x_{i}
            - root_norm: norm of roots, z_{i}
            - i: number of succeded iterations

    """
    # Initialize inputs
    step_norm = 1.0 + step_tol #initialize step norm
    root_norm = 1.0 + root_tol #initialize root norm
    i = 0 #initialize counter for number of iterations

    step_factor = 4.0
    step_min = 1.0
    step_max = 1.0

    if acceleration == "None":
        # Set up fixed-point equations
        funFXP = lambda x: UpdateFixedPointRoot(x,fun)

        # Set up stopping criterion function
        conditionFXP = ConditionFixedPointRoot

    elif acceleration == "SQUAREM":
        # Set up accelerated fixed-point equations (SQUAREM)
        funFXP = lambda x: UpdateSQUAREM(x,fun)

        # Set up stopping criterion function
        conditionFXP = ConditionSQUAREM

        

    out = (x0, step_norm, root_norm, step_tol, root_tol, max_iter, i, step_factor, step_min, step_max)

    out = funFXP(out)

    # Execute fixed-point iterations
    out = lax.while_loop(body_fun=funFXP,
                         cond_fun=conditionFXP,
                         init_val=out)

    # Unpack output tuple
    (x1, step_norm, root_norm, step_tol, root_tol, max_iter, i, step_factor, step_min, step_max) = out

    # Print number of iterations and norm of root and step
    print(f'FXP output: iterations={i}, root norm={root_norm}, step norm={step_norm}')

    return x1, (step_norm, root_norm, i)