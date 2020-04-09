#!/usr/bin/env python
# coding: utf-8

# As before, we can easily extend this projection operator to cases where the measure of distance between $\mathbf{y}$ and the subspace $\mathbf{v}$ is weighted (i.e. non-uniform). We can accomodate these weighted distances by re-writing the projection operator as
# 
# $$ \mathbf{P}_{V} = \mathbf{V} ( \mathbf{V}^T \mathbf{Q V})^{-1} \mathbf{V}^T \mathbf{Q} $$
# 
# where $\mathbf{Q}$ is positive definite matrix.  Earlier, we started with a point $\mathbf{y}$ and inflated a sphere  centered at $\mathbf{y}$ until it just touched the plane defined by $\mathbf{v}_i$ and this point was closest point on the subspace to $\mathbf{y}$. In the general case with a weighted distance except now we inflate an ellipsoid, not a sphere, until the ellipsoid touches the line.
# 
# The code and figure below illustrate what happens using the weighted $ \mathbf{P}_v $. It is basically the same code we used above. You can download the IPython notebook corresponding to this post and try different values on the diagonal of $\mathbf{Q}$.

# In[69]:


x = np.linspace(0,2,50)
y = x + x**2 + np.random.randn(len(x))

V = matrix(np.vstack([ones(x.shape),x,x**2]).T)
Q = np.matrix(np.eye(V.shape[0]))
i,j =np.diag_indices_from(Q)
Q[i[:20],j[:20]]=100
R = np.matrix(np.diag([1,1,13]))*0

Pv = V*inv(V.T*Q*V+R)*V.T*Q

p=np.polyfit(x,y,2)

fig, ax = subplots()
fig.set_size_inches(5,5)

ax.plot(x,y,'o',label='data',alpha=0.3)
ax.plot(x,np.dot(Pv,y).flat,label='projection')
ax.plot(x,np.polyval(p,x),'-',label='polyfit',alpha=0.3)
ax.grid()
ax.legend(loc=0)


# In[2]:


x = np.random.rand(50)*2
x.sort() # sort for plotting
       
y = x + x**2 + np.random.randn(len(x))

V = matrix(np.vstack([ones(x.shape),x,x**2]).T)
Pv = V*inv(V.T*V+eye(3)*4)*V.T

p=np.polyfit(x,y,2)

fig, ax = subplots()
fig.set_size_inches(5,5)

ax.plot(x,y,'o',label='data',alpha=0.3)
ax.plot(x,np.dot(Pv,y).flat,'-s',label='projection')
#ax.plot(x,np.polyval(p,x),'s',label='polyfit')
ax.grid()
ax.legend(loc=0);


# In[77]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider 

fig, ax = plt.subplots()
fig.set_size_inches(5,5)

plt.subplots_adjust(left=0.25, bottom=0.30,right=1)

x = np.linspace(0,2,50)
y = x + x**2 + np.sin(2*np.pi*x) + np.random.randn(len(x))

V = np.matrix(np.vstack([np.ones(x.shape),x,x**2]).T)
Q = np.matrix(np.eye(V.shape[0]))
i,j =np.diag_indices_from(Q)
#Q[i[:20],j[:20]]=100
R = np.matrix(np.diag([1,1,1]))*0

Pv = V*np.linalg.inv(V.T*Q*V+R)*V.T*Q
p=np.polyfit(x,y,2)

ax.plot(x,y,'o',label='data',alpha=0.3)
ax.set_title('err^2=%3.2f'%(np.linalg.norm(y)**2 - np.linalg.norm(np.dot(Pv,y))**2 ))
ls_line,=ax.plot(x,np.dot(Pv,y).flat,label='projection')
ax.plot(x,np.polyval(p,x),'-',label='polyfit',alpha=0.3)
ax.legend(loc=0)
ax.grid()

axw0 = plt.axes([0.25, 0.2, 0.65, 0.03] )
sw0  = Slider(axw0, 'w0', 0.1, 30.0, valinit = 1)
axw1 = plt.axes([0.25, 0.15, 0.65, 0.03] )
sw1  = Slider(axw1, 'w1', 0.1, 30.0, valinit = 1)
axw2 = plt.axes([0.25, 0.10, 0.65, 0.03] )
sw2  = Slider(axw2, 'w2', 0.1, 30.0, valinit = 1)

def update_w0(val):
   w = sw0.val
   R[0,0]=w
   Pv = V*np.linalg.inv(V.T*Q*V+R)*V.T*Q
   ls_line.set_ydata(np.dot(Pv,y).flat)
   ax.set_title('err=%3.2f'%(np.linalg.norm(y)**2 - np.linalg.norm(np.dot(Pv,y))**2 ))
   plt.draw()
sw0.on_changed(update_w0)

def update_w1(val):
   w = sw1.val
   R[1,1]=w
   Pv = V*np.linalg.inv(V.T*Q*V+R)*V.T*Q
   ls_line.set_ydata(np.dot(Pv,y).flat)
   ax.set_title('err=%3.2f'%(np.linalg.norm(y)**2 - np.linalg.norm(np.dot(Pv,y))**2 ))
   plt.draw()
sw1.on_changed(update_w1)

def update_w2(val):
   w = sw2.val
   R[2,2]=w
   Pv = V*np.linalg.inv(V.T*Q*V+R)*V.T*Q
   ls_line.set_ydata(np.dot(Pv,y).flat)
   ax.set_title('err=%3.2f'%(np.linalg.norm(y)**2 - np.linalg.norm(np.dot(Pv,y))**2 ))
   plt.draw()
sw2.on_changed(update_w2)

plt.show()


# ### References

# This post was created using the [nbconvert](https://github.com/ipython/nbconvert) utility from the source [IPython Notebook](www.ipython.org) which is available for [download](https://github.com/unpingco/Python-for-Signal-Processing/blob/master/Projection_mdim.ipynb) from the main github [site](https://github.com/unpingco/Python-for-Signal-Processing) for this blog. The projection concept is masterfully discussed in the classic Strang, G. (2003). *Introduction to linear algebra*. Wellesley Cambridge Pr. Also, some of Dr. Strang's excellent lectures are available on [MIT Courseware](http://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/). I highly recommend these as well as the book.
