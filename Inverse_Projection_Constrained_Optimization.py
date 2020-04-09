#!/usr/bin/env python
# coding: utf-8

# Inverse Projection
# -------------------------------------------------
# 
# In our [previous discussion](http://python-for-signal-processing.blogspot.com/2012/11/projection-in-multiple-dimensions-in.html), we developed a powerful intuitive sense for projections and their relationships to minimum mean squared error problems. Here, we continue to extract yet another powerful result from the same concept. Recall that we learned to minimize
# 
# $$ J = || \mathbf{y} - \mathbf{V}\boldsymbol{\alpha} ||^2  $$
# 
# by projecting $\mathbf{y}$ onto the space characterized by $\mathbf{V}$ as follows:
# 
# $$ \mathbf{\hat{y}} = \mathbf{P}_V \mathbf{y}$$
# 
# where
# 
# $$ \mathbf{P}_V  = \mathbf{V} (\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T$$
# 
# where the corresponding error ($\boldsymbol{\epsilon}$) comes directly from the Pythagorean theorem:
# 
# $$ ||\mathbf{y}||^2  = ||\mathbf{\hat{y}}||^2 + ||\boldsymbol{\epsilon}||^2 $$
# 
# Now, let's consider the inverse problem: given $\mathbf{\hat{y}}$, what is the corresponding $\mathbf{y}$? The first impulse in this situation is to re-visit 
# 
# $$ \mathbf{\hat{y}} = \mathbf{P}_V \mathbf{y}$$
# 
# and see if you can somehow compute the inverse of $\mathbf{P}_V$. This will not work because the projection matrix does not possess a unique inverse. In the following figure, the vertical sheet represents all vectors in the space that have exactly the same projection, $\mathbf{\hat{y}}$. Thus, there is no unique solution to the inverse problem which is to say that it is "ill-posed".

# In[1]:


#http://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot

from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
fig.set_size_inches([8,8])

ax = fig.add_subplot(111, projection='3d')

ax.set_aspect(1)
ax.set_xlim([0,2])
ax.set_ylim([0,2])
ax.set_zlim([0,2])
ax.set_aspect(1)
ax.set_xlabel('x-axis',fontsize=16)
ax.set_ylabel('y-axis',fontsize=16)
ax.set_zlabel('z-axis',fontsize=16)

y = matrix([1,1,1]).T 
V = matrix([[1,0.25], # columns are v_1, v_2
            [0,0.50],
            [0,0.00]])

alpha=inv(V.T*V)*V.T*y # optimal coefficients
P = V*inv(V.T*V)*V.T
yhat = P*y         # approximant


u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)

xx = np.outer(np.cos(u), np.sin(v))
yy = np.outer(np.sin(u), np.sin(v))
zz = np.outer(np.ones(np.size(u)), np.cos(v))

sphere=ax.plot_surface(xx+y[0,0], yy+y[1,0], zz+y[2,0],  
                       rstride=4, cstride=4, color='gray',alpha=0.3,lw=0.25)

ax.plot3D([y[0,0],0],[y[1,0],0],[y[2,0],0],'r-',lw=3)
ax.plot3D([y[0,0]],[y[1,0]],[y[2,0]],'ro')

ax.plot3D([V[0,0],0],[V[1,0],0],[V[2,0],0],'b-',lw=3)
ax.plot3D([V[0,0]],[V[1,0]],[V[2,0]],'bo')
ax.plot3D([V[0,1],0],[V[1,1],0],[V[2,1],0],'b-',lw=3)
ax.plot3D([V[0,1]],[V[1,1]],[V[2,1]],'bo')

ax.plot3D([yhat[0,0],0],[yhat[1,0],0],[yhat[2,0],0],'g--',lw=3)
ax.plot3D([yhat[0,0]],[yhat[1,0]],[yhat[2,0]],'go')


x2, y2, _ = proj3d.proj_transform(y[0,0],y[1,0],y[2,0], ax.get_proj())
ax.annotate(
    "$\mathbf{y}$", 
    xy = (x2, y2), xytext = (40, 20), fontsize=24,
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

x2, y2, _ = proj3d.proj_transform(yhat[0,0],yhat[1,0],yhat[2,0], ax.get_proj())
ax.annotate(
    "$\hat{\mathbf{y}}$", 
    xy = (x2, y2), xytext = (40, 10), fontsize=24,
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

x2, y2, _ = proj3d.proj_transform(V[0,0],V[1,0],V[2,0], ax.get_proj())
ax.annotate(
    "$\mathbf{v}_1$", 
    xy = (x2, y2), xytext = (120, 10), fontsize=24,
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

x2, y2, _ = proj3d.proj_transform(V[0,1],V[1,1],V[2,1], ax.get_proj())
ax.annotate(
    "$\mathbf{v}_2$", 
    xy = (x2, y2), xytext = (-30, 30), fontsize=24,
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

xx = array([0,yhat[0],yhat[0]])
yy = array([0,yhat[1],yhat[1]])
zz = array([0,0,2])

ax.add_collection3d( art3d.Poly3DCollection([zip(xx,yy,zz)],alpha=0.15,color='m') )
ax.set_title(r'The magenta sheet contains vectors with the same projection, $\mathbf{\hat{y}}$')

plt.show()


# Thus, since the unique inverse does not exist, we can impose constraints on the solution to enforce a unique solution. For example, since there are many $\mathbf{y}$ that correspond to the same  $\mathbf{\hat{y}}$, we can pick the one that has the shortest length, $||\mathbf{y} ||$. Thus, we can solve this inverse projection problem by enforcing additional constraints. 
# 
# Consider the following constrained minimization problem:
# 
# $$ \min_y \mathbf{y}^T \mathbf{y}$$
# 
# subject to:
# 
# $$ \mathbf{v_1}^T \mathbf{y} = c_1$$
# $$ \mathbf{v_2}^T \mathbf{y} = c_2$$
# 
# Here, we have the same setup as before: we want to minimize something with a set of constraints. We can re-write the constraints in a more familiar form as
# 
# $$ \mathbf{V}^T \mathbf{y} = \mathbf{c}$$
# 
# We can multiply both sides to obtain the even more familiar form:
# 
# $$ \mathbf{V} (\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T \mathbf{y} = \mathbf{V} (\mathbf{V}^T \mathbf{V})^{-1}\mathbf{c} $$
# 
# which by cleaning up the notation gives,
# 
# $$ \mathbf{P}_V \mathbf{y} = \mathbf{\hat{y}} $$
# 
# where
# 
# $$  \mathbf{\hat{y}}  = \mathbf{V} (\mathbf{V}^T \mathbf{V})^{-1}\mathbf{c}$$
# 
# and
# 
# $$ \mathbf{P}_V=\mathbf{V} (\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T $$
# 
# So far, nothing has really happened yet. We've just rewritten the constraints as a projection, but we still don't know how to find the $\mathbf{y}$ of minimum length. To do that, we turn to the Pythagorean relationship we pointed out earlier,
# 
# $$ ||\mathbf{y}||^2  = ||\mathbf{\hat{y}}||^2 + ||\boldsymbol{\epsilon}||^2 $$
# 
# where $\boldsymbol{\epsilon} = \mathbf{y} - \mathbf{\hat{y}}$. So, we have
# 
# $$ ||\mathbf{y}||^2  = ||\mathbf{\hat{y}}||^2 + || \mathbf{y} - \mathbf{\hat{y}}||^2 \ge 0$$
# 
# and since this is always non-negative, the only way to minimize $||\mathbf{y}||^2$ is to set 
# 
# $$ \mathbf{y} = \mathbf{\hat{y}} =  \mathbf{V} (\mathbf{V}^T \mathbf{V})^{-1}\mathbf{c}  $$
# 
# which is the solution to our constrained minimization problem.
# 
# If that seems shockingly easy, remember that we have already done all the heavy lifting by solving the projection problem. Here, all we have done is re-phrased the same problem.
# 
# Let's see this in action using `scipy.optimize` for comparison. Here's the problem
# 
# $$ \min_y \mathbf{y}^T \mathbf{y}$$
# 
# subject to:
# 
# $$ \mathbf{e_1}^T \mathbf{y} = 1$$
# $$ \mathbf{e_2}^T \mathbf{y} = 1$$
# 
# where $\mathbf{e}_i$ is the coordinate vector that is zero everywhere except for the $i^{th}$ entry.
# 

# In[2]:


import numpy as np
from scipy.optimize import minimize

# constraints formatted for scipy.optimize.minimize
cons = [{'type':'eq','fun':lambda x: x[0]-1,'jac':None},
        {'type':'eq','fun':lambda x: x[1]-1,'jac':None},
        ]

init_point = np.array([1,2,3,0]) # initial guess
ysol= minimize(lambda x: np.dot(x,x),init_point,constraints=cons,method='SLSQP')

# using projection method
c = np.matrix([[1,1,0,0]]).T     # RHS constraint vector
V = np.matrix(np.eye(4)[:,0:2])
ysol_p =  V*np.linalg.inv(V.T*V)*V.T*c

print 'scipy optimize solution:',
print ysol['x']
print 'projection solution:',
print np.array(ysol_p).flatten()

print np.allclose(np.array(ysol_p).flat,ysol['x'],atol=1e-6)


# ## Weighted Constrained Minimization

# We can again pursue a more general problem using the same technique:
# 
# $$ \min_y \mathbf{y}^T \mathbf{Q} \mathbf{y} $$
# 
# subject to:
# 
# $$ \mathbf{V}^T \mathbf{y} = \mathbf{c}$$
# 
# where $\mathbf{Q}$ is a positive definite matrix. In this case, we can define $\eta$ such that:
# 
# $$ \mathbf{y} = \mathbf{Q}^{-1} \boldsymbol{\eta}$$
# 
# and re-write the constraint as
# 
# $$ \mathbf{V}^T \mathbf{Q}^{-1} \boldsymbol{\eta} = \mathbf{c}$$ 
# 
# and multiply both sides,
# 
# $$\mathbf{P}_V \boldsymbol{\eta}=\mathbf{V} ( \mathbf{V}^T \mathbf{Q^{-1} V})^{-1} \mathbf{c} $$
# 
# where
# 
# $$\mathbf{P}_V=\mathbf{V} ( \mathbf{V}^T \mathbf{Q^{-1} V})^{-1} \mathbf{V}^T \mathbf{Q}^{-1}$$
# 
# To sum up, the minimal $\mathbf{y}$ that solves this constrained minimization problem is
# 
# $$ \mathbf{y}_o = \mathbf{V} ( \mathbf{V}^T \mathbf{Q^{-1} V})^{-1} \mathbf{c}$$
# 
# Once again, let's illustrate this using `scipy.optimize` in the following

# In[3]:


import numpy as np
from scipy.optimize import minimize

# constraints formatted for scipy.optimize.minimize
cons = [{'type':'eq','fun':lambda x: x[0]-1,'jac':None},
        {'type':'eq','fun':lambda x: x[1]-1,'jac':None},
        ]

Q = np.diag ([1,2,3,4])


init_point = np.array([1,2,3,0]) # initial guess
ysol= minimize(lambda x: np.dot(x,np.dot(Q,x)),init_point,constraints=cons,method='SLSQP')

# using projection method
Qinv = np.linalg.inv(Q)
c = np.matrix([[1,1,0,0]]).T     # RHS constraint vector
V = np.matrix(np.eye(4)[:,0:2])
ysol_p =  V*np.linalg.inv(V.T*Qinv*V)*V.T*Qinv*c
print 'scipy optimize solution:',
print ysol['x']
print 'projection solution:',
print np.array(ysol_p).flatten()
print np.allclose(np.array(ysol_p).flat,ysol['x'],atol=1e-5)


# ## Summary

# In this section, we pulled yet another powerful result from the projection concept we developed previously. We showed how "inverse projection" can lead to the solution of the classic constrained minimization problem. Although there are many approaches to the same problem, by once again appealing to the power projection method, we can maintain our intuitive geometric intuition. 

# ### References

# This post was created using the [nbconvert](https://github.com/ipython/nbconvert) utility from the source [IPython Notebook](www.ipython.org) which is available for [download](https://github.com/unpingco/Python-for-Signal-Processing/blob/master/Inverse_Projection_Constrained_Optimization.ipynb) from the main github [site](https://github.com/unpingco/Python-for-Signal-Processing) for this blog. The projection concept is masterfully discussed in the classic Strang, G. (2003). *Introduction to linear algebra*. Wellesley Cambridge Pr. Also, some of Dr. Strang's excellent lectures are available on [MIT Courseware](http://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/). I highly recommend these as well as the book.
