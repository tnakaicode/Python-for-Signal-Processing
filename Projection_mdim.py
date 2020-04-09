#!/usr/bin/env python
# coding: utf-8

# Projection in Multiple Dimensions
# -----------------------------------------
# 
#  this section, we extend from the [one-dimensional subspace](http://python-for-signal-processing.blogspot.com/2012/11/the-projection-concept.html) to the more general two-dimensional subspace. This np.means that there are two vectors,  $\mathbf{v}_1$ and  $\mathbf{v}_2$ that are not colinear and that span the subspace. In the previous case, we had only one vector ( $\mathbf{v}$), so we had a one-dimensional subspace, but now that we have two vectors, we have a two-dimensional subspace (i.e. a plane). The extension from the two-dimensional subspace to the *n*-dimensional subspace follows the same argument but introduces more notation than we need so we'll stick with the two-dimensional case for awhile. For the two-dimensional case, the optimal MMSE solution has the form
# 
# $$ \hat{\mathbf{y}} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 \in \mathbb{R}^m $$
# 
# where $\mathbf{y}$ exists in the m-dimensional space of real numbers. We want to project this vector onto the two m-dimensional  $\mathbf{v}_i$ vectors.  Here, the orthogonality requirement extends as
# 
# $$ \langle \mathbf{y} - \alpha_1 \mathbf{v}_1 -\alpha_2 \mathbf{v}_2 , \mathbf{v}_1\rangle= 0 $$ 
# 
# and 
# 
# $$ \langle \mathbf{y} - \alpha_1 \mathbf{v}_1 -\alpha_2 \mathbf{v}_2 , \mathbf{v}_2\rangle= 0 $$ 
# 
# Recall that for vectors, we have
# 
# $$ \langle \mathbf{x} , \mathbf{y}\rangle  = \mathbf{x}^T \mathbf{y} \in \mathbb{R}$$
# 
# This leads to the linear system of equations:
# 
# $$ \begin{eqnnp.array}
# \langle \mathbf{y}, \mathbf{v}_1\rangle = & \alpha_1 \langle \mathbf{v}_1, \mathbf{v}_1\rangle  &  +\alpha_2 \langle \mathbf{v}_1, \mathbf{v}_2\rangle   \\\\
# \langle \mathbf{y}, \mathbf{v}_2\rangle = & \alpha_1 \langle \mathbf{v}_1, \mathbf{v}_2\rangle  &  +\alpha_2 \langle \mathbf{v}_2, \mathbf{v}_2\rangle 
# \end{eqnnp.array}
# $$
# 
# which can be written in matrix form as
# 
# $$ \left[ \begin{np.array}{c}
# \langle \mathbf{y}, \mathbf{v}_1\rangle  \\\\
# \langle \mathbf{y}, \mathbf{v}_2\rangle  \\\\
# \end{np.array} \right] = 
# \left[ \begin{np.array}{cc}
# \langle \mathbf{v}_1, \mathbf{v}_1\rangle & \langle \mathbf{v}_1, \mathbf{v}_2\rangle \\\\
# \langle \mathbf{v}_1, \mathbf{v}_2\rangle  & \langle \mathbf{v}_2, \mathbf{v}_2\rangle \\\\
# \end{np.array} \right] \left[
# \begin{np.array}{c}
# \alpha_1  \\\\
# \alpha_2  \\\\
# \end{np.array} \right]$$ 
# 
# which can be further reduced by stacking the columns into 
# 
# $$ \mathbf{V} = \left[ \mathbf{v}_1, \mathbf{v}_2 \right] \in \mathbb{R}^{m \times 2} $$
# 
# and 
# 
# $$ \boldsymbol{\alpha}= \left[ \alpha_1, \alpha_2\right]^T \in \mathbb{R}^{2}$$
# 
# which gives
# 
# $$ \mathbf{V}^T \mathbf{y} = (\mathbf{V}^T \mathbf{V}) \boldsymbol{\alpha} $$
# 
# Note that by writing this using vector notation, we have implicitly generalized beyond two dimensions since there is nothing to stop from stacking $\mathbf{V}$ with more column vectors to create a larger subspace. By solving we obtain,
# 
# $$ \boldsymbol{\alpha} = (\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T \mathbf{y} $$ 
# 
# and so the optimal solution is then,
# 
# $$ \hat{\mathbf{y}} = \mathbf{V} \boldsymbol{\alpha} \in \mathbb{R}^m $$ 
# 
# Note that the existence of the inverse is guaranteed by the non-co-linearity of the $\mathbf{v}_i$ vectors. Whether or not that inverse is numerically stable is another issue.
# 
# Then, we can combine these to obtain
# 
# $$ \hat{\mathbf{y}} = \mathbf{V} (\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T \mathbf{y} $$ 
# 
# when then makes the projection operator for this case:
# 
# $$ \mathbf{P}_{V}= \mathbf{V} (\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T \in \mathbb{R}^{m \times m} $$ 
# 
# As a quick check, we can see this reduce to the 1-dimensional case by setting
# 
# $$ \mathbf{V}= \mathbf{v} \in \mathbb{R}^m$$
# 
# so then,
# 
# $$ \mathbf{P}_{v}= \mathbf{v} \frac{1}{\mathbf{v}^T \mathbf{v}} \mathbf{v}^T $$ 
# 
# which matches our [previous result](http://python-for-signal-processing.blogspot.com/2012/11/the-projection-concept.html). The point of all these manipulations is that we can construct another projection operator with all the MMSE properties we had before, but now in a bigger subspace.  We can further verify the idempotent property of projection matrices by checking that
# 
# $$ \mathbf{P}_V \mathbf{P}_V = \mathbf{P}_V$$
# 
# The following graphic shows that when we project the three dimensional $\mathbf{y}$ vector onto the plane, which is spanned by the two $\mathbf{v}_i$ vectors, we obtain the MMSE solution where the sphere is tangent to the plane. The point of tangency is the  point $\hat{\mathbf{y}} $ which is the MMSE solution.

# [2]


#http://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot

from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import numpy as np as np

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


u = np.np.linspace(0, 2*np.pi, 100)
v = np.np.linspace(0, np.pi, 100)

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
    xy = (x2, y2), xytext = (-20, 20), fontsize=24,
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

x2, y2, _ = proj3d.proj_transform(yhat[0,0],yhat[1,0],yhat[2,0], ax.get_proj())
ax.annotate(
    "$\hat{\mathbf{y}}$", 
    xy = (x2, y2), xytext = (-40, 10), fontsize=24,
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

plt.show()


# ### Weighted Distances

# As before, we can easily extend this projection operator to cases where the measure of distance between $\mathbf{y}$ and the subspace $\mathbf{v}$ is weighted (i.e. non-uniform). We can accomodate these weighted distances by re-writing the projection operator as
# 
# $$ \mathbf{P}_{V} = \mathbf{V} ( \mathbf{V}^T \mathbf{Q V})^{-1} \mathbf{V}^T \mathbf{Q} $$
# 
# where $\mathbf{Q}$ is positive definite matrix.  Earlier, we started with a point $\mathbf{y}$ and inflated a sphere  centered at $\mathbf{y}$ until it just touched the plane defined by $\mathbf{v}_i$ and this point was closest point on the subspace to $\mathbf{y}$. In the general case with a weighted distance except now we inflate an ellipsoid, not a sphere, until the ellipsoid touches the line.
# 
# The code and figure below illustrate what happens using the weighted $ \mathbf{P}_v $. It is basically the same code we used above. You can download the IPython notebook corresponding to this post and try different values on the diagonal of $\mathbf{Q}$.

# [9]


from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import numpy as np as np

fig = plt.figure()
fig.set_size_inches([8,8])

ax = fig.add_subplot(111, projection='3d')
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

Q = matrix([[1,0,0],
            [0,2,0],
            [0,0,3]])

P = V*inv(V.T*Q*V)*V.T*Q
yhat = P*y         # approximant


u = np.np.linspace(0, 2*np.pi, 100)
v = np.np.linspace(0, np.pi, 100)

xx = np.outer(np.cos(u), np.sin(v))
yy = np.outer(np.sin(u), np.sin(v))
zz = np.outer(np.ones(np.size(u)), np.cos(v))

xx,yy,yz=map(squeeze,split(tensordot(dstack([xx,yy,zz]),Q,axes=1),3,axis=2))

ellipsoid=ax.plot_surface(xx+y[0,0], yy+y[1,0], zz+y[2,0],  
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
    xy = (x2, y2), xytext = (-20, 20), fontsize=24,
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

x2, y2, _ = proj3d.proj_transform(yhat[0,0],yhat[1,0],yhat[2,0], ax.get_proj())
ax.annotate(
    "$\hat{\mathbf{y}}$", 
    xy = (x2, y2), xytext = (40, 30), fontsize=24,
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

plt.show()


# ## Summary

#  this section, we extended the concept of a projection operator beyond one dimension and showed the corresponding geometric concepts that tie the projection operator to MMSE problems in more than one dimension. 

# ### References

# This post was created using the [nbconvert](https://github.com/ipython/nbconvert) utility from the source [IPython Notebook](www.ipython.org) which is available for [download](https://github.com/unpingco/Python-for-Signal-Processing/blob/master/Projection_mdim.ipynb) from the main github [site](https://github.com/unpingco/Python-for-Signal-Processing) for this blog. The projection concept is masterfully discussed in the classic Strang, G. (2003). *Introduction to linear algebra*. Wellesley Cambridge Pr. Also, some of Dr. Strang's excellent lectures are available on [MIT Courseware](http://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/). I highly recommend these as well as the book.

# ### Appendix

# Below is some extra code to handle the more general case where there is a rotation as well as a weighting of the axes. Note the projection operator is constructed exactly the same way.

# [15]


from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import numpy as np as np

fig = plt.figure()
fig.set_size_inches([8,8])

ax = fig.add_subplot(111, projection='3d')
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

def rotation_matrix(angle,axis='z'):                  
    angle = angle/180.*pi
    if axis=='z':
        return matrix([[cos(angle),sin(angle),0],
                  [sin(-angle),cos(angle),0],
                  [0,0,1]])
    elif axis=='y':
        return matrix([[cos(angle),sin(angle),0],
                       [0,1,0],
                       [sin(-angle),cos(angle),0]])
    elif axis=='x':
        return matrix([[1,0,0],
                       [cos(angle),sin(angle),0],
                       [sin(-angle),cos(angle),0]])
          
S = matrix([[3,0,0],
            [0,2,0],
            [0,0,1]])

R = rotation_matrix(30)*rotation_matrix(30,'x')*rotation_matrix(40,'y')

Q = R.T*S*R # apply 3-D rotations

P = V*inv(V.T*Q*V)*V.T*Q # build projection matrix
yhat = P*y         # approximant


u = np.np.linspace(0, 2*np.pi, 100)
v = np.np.linspace(0, np.pi, 100)

xx = np.outer(np.cos(u), np.sin(v))
yy = np.outer(np.sin(u), np.sin(v))
zz = np.outer(np.ones(np.size(u)), np.cos(v))

xx,yy,yz=map(squeeze,split(tensordot(dstack([xx,yy,zz]),Q,axes=1),3,axis=2))

ellipsoid=ax.plot_surface(xx+y[0,0], yy+y[1,0], zz+y[2,0],  
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
    xy = (x2, y2), xytext = (-20, 20), fontsize=24,
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

x2, y2, _ = proj3d.proj_transform(yhat[0,0],yhat[1,0],yhat[2,0], ax.get_proj())
ax.annotate(
    "$\hat{\mathbf{y}}$", 
    xy = (x2, y2), xytext = (40, 30), fontsize=24,
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

plt.show()

