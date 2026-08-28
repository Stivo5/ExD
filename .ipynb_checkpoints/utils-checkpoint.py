import numpy as np


# Sign function
def sgn(op1, op2):
    '''
    Compute the relative statistics between two operators

    op: a pair
        (operator, statistc)
    '''

    if op1[1] == 'Fermion' and op2[1] == 'Fermion':
        sgn = -1
    else:
        sgn = 1
    return sgn



# Basic function (and its derivatives) for which divided difference in computed
def exp_int(x, beta):
    if np.abs(x) > 1e-8:
        f = np.expm1(x * beta)/x
    else:
        f = beta
    return f

def exp_int_d1x(x, beta):
    if np.abs(x) > 1e-8:
        f = beta * np.exp(x * beta)/x - np.expm1(x * beta) / x**2
    else:
        f = beta**2 / 2
    return f

def exp_int_d2x(x, beta):
    if np.abs(x) > 1e-8:
        f = beta**2 * np.exp(x * beta)/x - 2*beta * np.exp(x * beta)/x**2 + 2*np.expm1(x * beta) / x**3
    else:
        f = beta**3 / 3
    return f



# Compute n-th order divided difference of the function exp_int
# 1st order divided difference, for 3-point correlation
def div_diff_exp_int_1ord(x, beta):
    '''
    Compute the divided difference given a list of values
    
    x: length-2 list on which to compute divided difference
    '''

    if abs(x[0] - x[1]) > 1e-8:
        div_diff = exp_int(x[0], beta)/(x[0]-x[1]) + exp_int(x[1], beta)/(x[1]-x[0])
    else:
        div_diff = exp_int_d1x(x[0], beta)

    return div_diff
    

