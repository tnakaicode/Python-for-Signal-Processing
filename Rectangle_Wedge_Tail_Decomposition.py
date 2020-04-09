#!/usr/bin/env python
# coding: utf-8

# [1]


from __future__ import  division
get_ipython().run_line_magic('qtconsole', '')


# [2]


import sympy as S
import sympy.stats as st


# [20]


u=S.symbols('u')
p=st.LogNormal('u',0,0.5)
pu=st.density(p)(u)
fx=S.lambdify(u,pu,'numpy')
x = np.linspace(0.001,4,100)
hist([st.sample(p) for i in range(1000)],bins=50,normed=1);
plot(x,fx(x),'r-',lw=3.)


# [162]


pw=[i*(S.Heaviside(u-j/2)-S.Heaviside(u-j/2-1/2)) for i,j in zip(S.symbols('a:8'),range(8))]
pwf = lambda i: sum(pw).subs(u,i)


# ## Rectangle Wedge Tail Decomposition

# 
# $$ f(x) = \exp\left(-\frac{(x-1)^2}{2x} \right)(x+1)/12 $$ 
# 
# where $x>0$

# [13]


x = np.linspace(0.001,10,100)
f= lambda x: (np.sqrt(1/x) + np.sqrt(x))/ (2.*pow(exp(1),pow(-np.sqrt(1/x) + np.sqrt(x),2)/ 2.)*np.sqrt(2*pi)*x)
fx = f(x)
plot(x,fx)


# [14]


u=S.symbols('u')
p=S.Piecewise(*[(f(j),(i<u<=j)) for i,j in zip(range(10),range(1,11))]+[(0,True)])
plot(x,[p.subs(u,i) for i in x],x,fx)


# [14]





# [ ]




