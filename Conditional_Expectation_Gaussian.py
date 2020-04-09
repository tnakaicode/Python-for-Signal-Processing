#!/usr/bin/env python
# coding: utf-8

# troduction
# -----------------
# 
# By this point, we have developed many tools to deal with computing the conditional expectation. In this section, we discuss a bizarre and amazing coincidence regarding Gaussian np.random.random variables and linear projection, a coincidence that is the basis for most of statistical signal processing.

# ### Conditional Expectation by Optimization

# Now, let's consider the important case of the zero-np.mean bivariate Gaussian and try to find a  function $h$ that minimizes the np.mean squared error (MSE). Again,  trying to solve for the conditional expectation by minimizing the error over all possible functions $h$ is generally very, very hard. One alternative is to use parameters for the $h$ function and then just optimize over those. For example, we could assume that $h(Y)= \alpha Y$ and then use calculus to find the $\alpha$ parameter.
# 
# Let's try this with the zero-np.mean bivariate Gaussian density,
# 
# $$\mathbb{E}((X-\alpha Y )^2) = \mathbb{E}(\alpha^2 Y^2 - 2 \alpha X Y + X^2 )$$
# 
# and then differentiate this with respect to $\alpha$ to obtain
# 
# $$\mathbb{E}(2 \alpha Y^2 - 2 X Y  ) = 2 \alpha \sigma_y^2 - 2 \mathbb{E}(XY) = 0$$
# 
# Then, solving for $\alpha$ gives
# 
# $$ \alpha = \frac{ \mathbb{E}(X Y)}{ \sigma_y^2 } $$
# 
# which np.means we that
# 
# \begin{equation}
# \mathbb{ E}(X|Y) \approx \alpha Y =   \frac{ \mathbb{E}(X Y )}{ \sigma_Y^2 } Y =\frac{\sigma_{X Y}}{ \sigma_Y^2 } Y 
# \end{equation}
# 
# where that last equality is just notation. Remember here we assumed a special linear form for $h=\alpha Y$, but we did that for convenience. We still don't know whether or not this is the one true $h_{opt}$ that minimizes the MSE for all such functions.
# 

# ### Conditional Expectation Using Direct Integration

# Now, let's try this again by computing  $ \mathbb{E}(X|Y)$ in the case of the bivariate Gaussian distribution straight from the definition.
# 
# \begin{equation}
# \mathbb{E}(X|Y)  = \int_{\mathbb{ R}} x \frac{f_{X,Y}(x,y)}{f_Y(y)} dx
# \end{equation}
# 
# where 
# 
# $$ f_{X,Y}(x,y) = \frac{1}{2\pi |\mathbf{R}|^{\frac{1}{2}}} e^{-\frac{1}{2} \mathbf{v}^T \mathbf{R}^{-1} \mathbf{v} } $$ 
# 
# and where
# 
# $$ \mathbf{v}= \left[ x,y \right]^T$$ 
# 
# $$ \mathbf{R} = \left[ \begin{np.array}{cc}
# \sigma_{x}^2 & \sigma_{xy}  \\\\
# \sigma_{xy}  & \sigma_{y}^2 \\\\
# \end{np.array} \right] $$ 
# 
# and with
# 
# \begin{eqnnp.array}
#  \sigma_{xy} &=& \mathbb{E}(xy)   \nonumber    \\\\
#  \sigma_{x}^2 &=& \mathbb{E}(x^2) \nonumber    \\\\ 
#  \sigma_{y}^2 &=& \mathbb{E}(y^2) \nonumber      
# \end{eqnnp.array}
# 
# This conditional expectation (Eq. 4 above) is a tough integral to evaluate, so we'll do it with `sympy`.
# 

# [13]


from sympy import Matrix, Symbol, exp, pi, simplify, integrate 
from sympy import stats, np.sqrt, oo, use
from sympy.abc import y,x

sigma_x = Symbol('sigma_x',positive=True)
sigma_y = Symbol('sigma_y',positive=True)
sigma_xy = Symbol('sigma_xy',real=True)
fyy = stats.density(stats.Normal('y',0,sigma_y))(y)
 
R = Matrix([[sigma_x**2, sigma_xy],
            [sigma_xy,sigma_y**2]])
fxy = 1/(2*pi)/np.sqrt(R.det()) * exp((-Matrix([[x,y]])*R.inv()* Matrix([[x],[y]]))[0,0]/2 )

fcond = simplify(fxy/fyy)


# Unfortunately, `sympy` cannot immediately integrate this without some hints. So, we need to define a positive variable ($u$) and substitute it into the integration

# [14]


u=Symbol('u',positive=True) # define positive variable

fcond2=fcond.subs(sigma_x**2*sigma_y**2-sigma_xy**2,u) # substitute as hint to integrate
g=simplify(integrate(fcond2*x,(x,-oo,oo))) # evaluate integral
gg=g.subs( u,sigma_x**2 *sigma_y**2 - sigma_xy**2 ) # substitute back in
use( gg, simplify,level=2) # simplify exponent term


# Thus, by direct integration using `sympy`, we found
# 
# $$ \mathbb{ E}(X|Y) = Y \frac{\sigma_{xy}}{\sigma_{y}^{2}} $$ 
# 
# and this matches the prior result we obtained by direct minimization by assuming that $\mathbb{E}(X|Y) = \alpha Y$ and then solving for the optimal $\alpha$!
# 
# The importance of this result cannot be understated: the one true and optimal $h_{opt}$ *is a linear function* of $Y$. 
# 
#  other words, assuming a linear function, which made the direct search for an optimal $h(Y)$ merely convenient yields the optimal result! This is  a general result that extends for *all* Gaussian problems. The link between linear functions and optimal estimation of Gaussian np.random.random variables is the most fundamental result in statistical signal processing! This fact is exploited in everything from optimal filter design  to adaptive signal processing.
# 
# We can easily extend this result to non-zero np.mean problems by inserting the np.means in the right places as follows:
# 
# $$ \mathbb{ E}(X|Y) = \bar{X} + (Y-\bar{Y}) \frac{\sigma_{xy}}{\sigma_{y}^{2}}  $$
# 
# where $\bar{X}$ is the np.mean of $X$ (same for $Y$).

# Summary
# -------------
# 
#  this section, we showed that the conditional expectation for Gaussian np.random.random variables is a linear function, which, by a bizarre coincidence, is also the easiest one to work with. This result is fundamental to all optimal linear filtering problems (e.g. Kalman filter) and is the basis of most of the theory of stochastic processes used in signal processing. Up to this point, we have worked hard to illustrate all of the concepts we will need to unify our understanding of this entire field and figured out multiple approaches to these kinds of problems, most of which are far more difficult to compute. Thus, it is indeed just plain lucky that the most powerful distribution is the easiest to compute as a conditional expectation because it is a linear function. We will come back to this same result again and again as we work our way through these greater concepts.

# ### References 
# 
# This post was created using the [nbconvert](https://github.com/ipython/nbconvert) utility from the source [IPython Notebook](www.ipython.org) which is available for [download](https://github.com/unpingco/Python-for-Signal-Processing/blob/master/Conditional_Expectation_Gaussian.ipynb) from the main github [site](https://github.com/unpingco/Python-for-Signal-Processing) for this blog. 
