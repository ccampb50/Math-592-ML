## HW4
## COSC 479
## Machine Learning
## Craig Campbell

import tensorflow
import numpy as np
import matplotlib.pyplot as plt

## GDM x_0=-5: Minimum at -0.0016620515246770736 - stuck at local minima
## GDM x_0=5:  Minimum at 3.3945614935369486
## Momentum x_0=-5: Minimum at 3.3945370369561223
## Momentum x_0=5:  Minimum at 3.39457213161253
## f(x) = x**6 - 4*x**5 - 2*x**3 + 3*x**2
## local mins at x=0 and x=3.95


def df(x):
    return 6*x**5 - 20*x**4 - 6*x**2 + 6*x
x_0 = -5 #begining value
next_x = x_0  # We start the search at x=3
next_v = 0.9
mu = 0.9 #momentum
delta = 0.0001  # learning rate
precision = 0.000001  # Desired precision of result
max_iters = 100000  # Maximum number of iterations

for i in range(max_iters):
    current_v = next_v
    current_x = next_x
    next_v = mu*current_v - delta * df(current_x+mu*current_v)
    next_x = current_x + next_v
    step = next_x - current_x
    if abs(step) <= precision:
        break
print("Minimum at", next_x)


####SANN#####
def f(x):
    return x**6 - 5*x**4 - 2*x**3 + 3*x**2

def h(x):
    if x<-6 or x>6:
        y=0
    else:
        y = f(x)
    return y
hv = np.vectorize(h) #vectorize the function
X= np.linspace(-6, 6, num=100)
plt.plot(X, f(X))
plt.show()
##########################
def SA(search_space, func, T):
    scale=np.sqrt(T)
    start=np.random.choice(search_space)
    x=start
    cur=func(x)
    history =[x]
    for i in range(2000):
        prop=x + np.random.normal()*scale#unif(-1,1)*scale
        if prop > 1 or prop <0 or np.log(np.random.rand()) *T >= -(func(prop)-cur):
        #if prop > 1 or prop <0 or np.random.rand() > np.exp(-(func(prop)-cur)/T):
            prop = x
        x=prop
        cur = func(x)
        T=0.90*T #reduce Temperature by 8%
        history.append(x)
    return x, history
np.linspace(-1, 1, num=1000)
x1, history = SA(X, h, T=10)
plt.plot(X, hv(X))
plt.scatter(x1, hv(x1), marker='X')
plt.plot(history, hv(history))
plt.show()
print(history[-1])
