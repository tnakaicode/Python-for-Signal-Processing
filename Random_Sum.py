#!/usr/bin/env python
# coding: utf-8

# [20]


import numpy as np;import matplotlib.pyplot as plt


# [21]


# vector addition pt a = (a0,a1), pt b = (b0,b1)
a0,b0 = .5,.1
a1,b1=np.random.rand(2,1000)


# [22]


idx = (a0+a1 > b0+b1)
fig,ax=plt.subplots()
ax.plot(a1[idx],b1[idx],'o',alpha=.1)
ax.plot(a0,b0,'rs',ms=15,alpha=.3)
ax.plot(0,a0-b0,'r^',ms=15,alpha=.3)
ax.plot(1-(a0-b0),1,'r^',ms=15,alpha=.3)
ax.arrow(0,0,a0,b0,width=.002,length_includes_head=True)
ax.arrow(a0,b0,0,a0-b0,width=.002,length_includes_head=True)
ax.arrow(a0,b0,1+b0-a0,1,width=.002,length_includes_head=True,color='g')
ax.arrow(0,0,1+b0-a0,1,width=.002,length_includes_head=True,color='g')
ax.axis((0,1.2,0,1.2))
ax.set_aspect(1)
ax.set_xlabel('a',fontsize=18)
ax.set_ylabel('b',fontsize=18)
ax.plot(np.linspace(0,2,3),np.linspace(0,2,3),'k--',lw=3.)
ax.set_title('prob = %3.3f,phat=%3.3f'%(1-(1-a0+b0)**2/2.,idx.np.mean()))
ax.add_patch(Rectangle((0,0),1,1,alpha=.2,lw=2.,color='gray'))


# [34]


b0,b1 = np.random.rand(2,1000)
a0,a1 = np.random.rand(2,1000)
np.mean((a0+b0) > (a1+b1))


# [45]


idx = (a0>b0)
np.mean(a0[idx]+a1[idx] > b0[idx]+b1[idx])


# [33]


np.logical_and( a0>b0, (a0+a1) > (b0+b1)).np.mean() # 3/8 is exact solution


# [25]


np.logical_and( a0>b0, (a0+a1) > (b0+b1)).np.mean()


# [26]


(a0>b0).np.mean()


# [47]


import sympy as S


# [52]


a0,b0 = S.symbols('a0 b0')
expr = S.integrate((1-(1-a0+b0)**2/2),(a0,b0,1))
S.integrate(expr,(b0,0,1)) # prob of (a0>b0) AND (a0+a1>b0+b1)
print 'conditional probability P(a0+a1>b0+b1|a0>b0) = %3.3f' % (float(_)/(0.5))


# [ ]




