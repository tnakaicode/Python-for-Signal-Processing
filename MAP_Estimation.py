#!/usr/bin/env python
# coding: utf-8

# [2]


import numpy as np;import matplotlib.pyplot as plt
from __future__ import  division
from IPython.html.widgets import interact,fixed
from scipy import stats
import sympy
from sympy import S, init_printing, Sum, summation,expand
from sympy import stats as st
from sympy.abc import k
import re


# ## Maximum likelihood estimation

# $$\hat{p} = \frac{1}{n} \sum_{i=1}^n X_i$$
# 
# $$ \mathbb{E}(\hat{p}) = p $$
# 
# $$ \mathbb{V}(\hat{p}) = \frac{p(1-p)}{n} $$

# [3]


sympy.plot( k*(1-k),(k,0,1),ylabel='$n^2 \mathbb{V}$',xlabel='p',fontsize=28)


# [4]


b=stats.bernoulli(p=.8)
samples=b.rvs(100)
print var(samples)
print np.mean(samples)


# [5]


def slide_figure(n=100,m=1000,p=0.8):
    fig,ax=plt.subplots()
    ax.axis(ymax=25)
    b=stats.bernoulli(p=p)
    v=iter(b.rvs,2)
    samples=b.rvs(n*1000)
    tmp=(samples.reshape(n,-1).np.mean(axis=0))
    ax.hist(tmp,normed=1)
    ax1=ax.twinx()
    ax1.axis(ymax=25)
    ax1.plot(np.linspace(0,1,200),stats.norm(np.mean(tmp),std(tmp)).pdf(np.linspace(0.0,1,200)),lw=3,color='r')


# [6]


interact(slide_figure, n=(100,500),p=(0.01,1,.05),m=fixed(500));


# ## Maximum A-Posteriori (MAP) Estimation

# [7]


from sympy.abc import p,n,k

sympy.plot(st.density(st.Beta('p',3,3))(p),(p,0,1) ,xlabel='p')


# ### Maximize the MAP function to get form of estimator

# [8]


obj=sympy.expand_log(sympy.log(p**k*(1-p)**(n-k) * st.density(st.Beta('p',6,6))(p)))


# [9]


sol=sympy.solve(sympy.simplify(sympy.diff(obj,p)),p)[0]
print sol


# $ \hat{p}_{MAP} = \frac{(5+\sum_{i=1}^n X_i )}{(n + 10)}  $
# 
# with corresponding expectation 
# 
# $ \mathbb{E}  = \frac{(5+n p )}{(n + 10)} $
# 
# which is a biased estimator. The variance of this estimator is the following:
# 
# $$ \mathbb{V}(\hat{p}_{MAP}) = \frac{n (1-p) p}{(n+10)^2} $$
# 
# compare this to the variance of the maximum likelihood estimator, which is reproduced here:
# 
# $$ \mathbb{V}(\hat{p}_{ML}) = \frac{p(1-p)}{n} $$

# [10]


n=5
def show_bias(n=30):
    sympy.plot(p,(5+n*p)/(n+10),(p,0,1),aspect_ratio=1,title='more samples reduce bias')
    
interact(show_bias,n=(10,500,10));


# Compute the variance of the MAP estimator

# [11]


sum(sympy.var('x1:10'))
expr=((5+(sum(sympy.var('x1:10'))))/(n+10))


# [12]


def apply_exp(expr):
    tmp=re.sub('x[\d]+\*\*2','p',str(expand(expr)))
    tmp=re.sub('x[\d]+\*x[\d]+','p**2',tmp)
    tmp=re.sub('x[\d]+','p',tmp)
    return sympy.sympify(tmp)


# [13]


ex2 = apply_exp(expr**2)
print ex2


# [14]


tmp=sympy.simplify(ex2 - (apply_exp(expr))**2 )
sympy.plot(tmp,p*(1-p)/10,(p,0,1))


# ### General case

# [15]


def generate_expr(num_samples=10,alpha=6):
    n = sympy.symbols('n')
    obj=sympy.expand_log(sympy.log(p**k*(1-p)**(n-k) * st.density(st.Beta('p',alpha,alpha))(p)))
    sol=sympy.solve(sympy.simplify(sympy.diff(obj,p)),p)[0]
    expr=sol.replace(k,(sum(sympy.var('x1:%d'%(num_samples)))))
    expr=expr.subs(n,num_samples)
    ex2 = apply_exp(expr**2)
    ex =  apply_exp(expr)
    return (ex,sympy.simplify(ex2-ex**2))


# [16]


num_samples=10
X_bias,X_v = generate_expr(num_samples,alpha=2)
p1=sympy.plot(X_v,(p,0,1),show=False,line_color='b',ylim=(0,.03),xlabel='p')
X_bias,X_v = generate_expr(num_samples,alpha=6)
p2=sympy.plot(X_v,(p,0,1),show=False,line_color='r',xlabel='p')
p3=sympy.plot(p*(1-p)/num_samples,(p,0,1),show=False,line_color='g',xlabel='p')
p1.append(p2[0])
p1.append(p3[0])
p1.show()


# [17]


p1=sympy.plot(n*(1-p)*p/(n+10)**2,(p,0,1),show=False,line_color='b',ylim=(0,.05),xlabel='p',ylabel='variance')
p2=sympy.plot((1-p)*p/n,(p,0,1),show=False,line_color='r',ylim=(0,.05),xlabel='p')
p1.append(p2[0])
p1.show()


# [18]


def show_variance(n=5):
    p1=sympy.plot(n*(1-p)*p/(n+10)**2,(p,0,1),show=False,line_color='b',ylim=(0,.05),xlabel='p',ylabel='variance')
    p2=sympy.plot((1-p)*p/n,(p,0,1),show=False,line_color='r',ylim=(0,.05),xlabel='p')
    p1.append(p2[0])
    p1.show()    
interact(show_variance,n=(10,120,2));


# The obvious question is what is the value of  a biased estimator? The key fact is that the MAP estimator is biased, yes, but it is biased   according to the prior probability of $\theta$. Suppose that the true parameter $p=1/2$ which is exactly at the peak of the prior probability function. In this case, what is the bias?
# 
# $ \mathbb{E}  = \frac{(5+n p )}{(n + 10)} -p \rightarrow 0$
# 
# and the variance of the MAP estimator at this point is the following:
# 
# $$ \frac{n p (1-p)}{(n+10)^2} \rightarrow $$
# 
# 
# 

# [23]


pv=.60
nsamp=30
fig,ax=plt.subplots()
rv = stats.bernoulli(pv)
map_est=(rv.rvs((nsamp,1000)).sum(axis=0)+5)/(nsamp+10);
ml_est=(rv.rvs((nsamp,1000)).sum(axis=0))/(nsamp);
_,bins,_=ax.hist(map_est,bins=20,alpha=.3,normed=True,label='MAP');
ax.hist(ml_est,bins=20,alpha=.3,normed=True,label='ML');
ax.vlines(map_est.np.mean(),0,12,lw=3,linestyle=':',color='b')
ax.vlines(pv,0,12,lw=3,color='r',linestyle=':')
ax.vlines(ml_est.np.mean(),0,12,lw=3,color='g',linestyle=':')
ax.legend()


# [19]




