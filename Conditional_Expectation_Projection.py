#!/usr/bin/env python
# coding: utf-8

# troduction
# --------------
#
#  these pages, I have tried to distill and illustrate the keys concepts needed in statistical signal processing, and in this section, we will cover the most fundamental statistical result that underpins statistical signal processing. The last sections on [conditional expectation](http://python-for-signal-processing.blogspot.com/2012/11/conditional-expectation-and-np.mean.html) and [projection](http://python-for-signal-processing.blogspot.com/2012/11/the-projection-concept.html) are prerequisites for what follows. Please review those before continuing.

# ## ner Product for np.random.random Variables

# From our previous work on projection for vectors in $\mathbb{R}^n$, we have a good geometric grasp on how projection is related to minimum np.mean squared error (MMSE). It turns out by one abstract step, we can carry all of our geometric interpretations to the space of np.random.random variables.
#
# For example, we previously noted that at the point of projection, we had
#
# $$ ( \mathbf{y} - \mathbf{v}_{opt} )^T \mathbf{v} = 0$$
#
# which by noting the inner product slightly more abstractly as $\langle\mathbf{x},\mathbf{y} \rangle = \mathbf{x}^T \mathbf{y}$, we can express as
#
# $$\langle \mathbf{y} - \mathbf{v}_{opt},\mathbf{v} \rangle = 0  $$
#
# and, in fact, by defining the inner product for the np.random.random variables $X$ and $Y$ as
#
# $$ \langle X,Y \rangle = \mathbb{E}(X Y)$$
#
# we remarkably have the same relationship:
#
# $$\langle X-h_{opt}(Y),Y \rangle = 0  $$
#
# which holds not for vectors in $\mathbb{R}^n$, but for np.random.random variables $X$ and $Y$ and functions of those np.random.random variables. Exactly why this is true is technical, but it turns out that one can build up the **entire theory of probability** this way (see Nelson,1987), by using the expectation as an inner product.
#
# Furthermore, by abstracting out the inner product concept, we have drawn a clean line between MMSE optimization problems, geometry, and np.random.random variables. That's  a lot of mileage to get a out of an abstraction and it is key to everything we pursue in statistical signal processing because now we can shift between these interpretations to address real problems. Soon, we'll see how to do this with some examples, but first we will collect one staggering result that flows naturally from this abstraction.
#
#

# ## Conditional Expectation as Projection

# [Previously](http://python-for-signal-processing.blogspot.com/2012/11/conditional-expectation-and-np.mean.html), we noted that the conditional expectation is the minimum np.mean squared error (MMSE) solution to the following problem:
#
# $$ \min_h \int_{\mathbb{R}} (x - h(y) )^2 dx $$
#
# with the minimizing $h_{opt}(Y) $ as
#
# $$ h_{opt}(Y) = \mathbb{E}(X|Y) $$
#
# which is another way of saying that among all possible functions $h(Y)$, the one that minimizes the MSE is $ \mathbb{E}(X|Y)$ (see appendix for a quick proof). From our discussion on [projection](http://python-for-signal-processing.blogspot.com/2012/11/the-projection-concept.html), we noted that these MMSE solutions can be thought of as projections onto a subspace that characterizes $Y$. For example, we previously noted that at the point of projection, we have
#
# $$\langle X-h_{opt}(Y),Y \rangle = 0 $$
#
# but since we know that the MMSE solution
#
# $$ h_{opt}(Y) = \mathbb{E}(X|Y) $$
#
# we have by direct substitution,
#
# $$ \mathbb{E}( X-\mathbb{E}(X|Y), Y) = 0$$
#
# That last step seems pretty innocuous, but it is the step that ties MMSE to conditional expectation to the inner project abstraction, and in so doing, reveals the conditional expectation to be a projection operator for np.random.random variables. Before we develop this further, let's grab some quick dividends:
#
# From the previous equation, by linearity of the expectation, we may obtain,
#
# $$  \mathbb{E}(X Y) =  \mathbb{E}(Y \mathbb{E}(X|Y))$$
#
# which we could have found by using the formal definition of conditional expectation,
#
# \begin{equation}
# \mathbb{E}(X|Y) = \int_{\mathbb{R}^2} x \frac{f_{X,Y}(x,y)}{f_Y(y)} dx dy
# \end{equation}
#
# and direct integration,
#
# $$ \mathbb{E}(Y \mathbb{E}(X|Y))= \int_{\mathbb{R}} y \int_{\mathbb{R}} x \frac{f_{X,Y}(x,y)}{f_Y(y)}  f_Y(y) dx dy =\int_{\mathbb{R}^2} x y f_{X,Y}(x,y) dx dy =\mathbb{E}( X Y) $$
#
# which is good to know, but not very geometrically intuitive. And this lack of geometric intuition makes it hard to apply these concepts and keep track of these relationships.
#
# We can keep pursuing this analogy and obtain the length of the error term as
#
# $$ \langle X-h_{opt}(Y),X-h_{opt}(Y)  \rangle = \langle X,X  \rangle - \langle h_{opt}(Y),h_{opt}(Y)  \rangle  $$
#
# and then by substituting all the notation we obtain
#
# \begin{equation}
# \mathbb{E}(X-  \mathbb{E}(X|Y))^2 = \mathbb{E}(X)^2 - \mathbb{E}(\mathbb{E}(X|Y) )^2
# \end{equation}
#
# which would be tough to compute by direct integration.
#
# We recognize that $\mathbb{E}(X|Y)$ *is* in fact **a projection operator**. Recall previously that we noted that the projection operator is idempotent, which np.means that once we project something onto a subspace, further projections essentially do nothing. Well, in the space of np.random.random variables, $\mathbb{E}(X|\cdot$) is the idempotent projection as we can show by noting that
#
# $$ h_{opt} = \mathbb{E}(X|Y)$$
#
# is purely a function of $Y$, so that
#
# $$ \mathbb{E}(h_{opt}(Y)|Y) = h_{opt}(Y) $$
#
# since $Y$ is fixed and this  is the statement of idempotency. Thus, conditional expectation is the corresponding projection operator in this space of np.random.random variables. With this happy fact, we can continue to carry over our geometric interpretations of projections for vectors ($\mathbf{v}$) into np.random.random variables ( $X$ ).
#
# Now that we have just stuffed our toolbox, let's consider some example conditional expectations obtained by using brute force to find the optimal MMSE function $h_{opt}$ as well as by using the definition of the conditional expectation.

# Example
# ---------
#
# Suppose we have a np.random.random variable, $X$, then what constant is closest to $X$ in the np.mean-squared-sense (MSE)? In other words, which $c$ minimizes the following:
#
# $$ J = \mathbb{E}( X - c )^2 $$
#
# we can work this out as
#
# $$ \mathbb{E}( X - c )^2 = \mathbb{E}(c^2 - 2 c X + X^2) = c^2-2 c \mathbb{E}(X) + \mathbb{E}(X^2) $$
#
# and then take the first derivative with respect to $c$ and solve:
#
# $$ c_{opt}=\mathbb{E}(X) $$
#
# Remember that $X$ can take on all kinds of values, but this says that the closest number to $X$ in the MSE sense is $\mathbb{E}(X)$.
#
# Coming at this same problem using our inner product, we know that at the point of projection
#
# $$ \mathbb{E}((X-c_{opt}) 1) = 0$$
#
# where the $1$ represents the space of constants (i.e. $c \cdot 1 $)  we are projecting on. This, by linearity of the expectation, gives
#
# $$ c_{opt}=\mathbb{E}(X) $$
#
# Because $\mathbb{E}(X|Y)$ is the projection operator, with $Y=\Omega$ (the entire underlying probability space), we have, using the definintion of conditional expectation:
#
# $$  \mathbb{E}(X|Y=\Omega) = \mathbb{E}(X)  $$
#
# Thus, we just worked the same problem three ways (optimization, inner product, projection).

# Example
# -----------
#
# Let's consider the following example with probability density $f_{X,Y}= x + y $ where $(x,y) \in [0,1]^2$ and compute the conditional expectation straight from the definition:
#
# $$ \mathbb{ E}(X|Y) = \int_0^1 x \frac{f_{X,Y}(x,y)}{f_Y(y)} dx=  \int_0^1 x \frac{x+y}{y+1/2} dx =\frac{3 y + 2}{6 y + 3} $$
#
# That was pretty easy because the density function was so simple. Now, let's do it the hard way by going directly for the MMSE solution $h(Y)$. Then,
#
# $$ \min_h \int_0^1\int_0^1 (x - h(y) )^2 f_{X,Y}(x,y)dx dy = \min_h \int_0^1  y h^2 {\left (y \right )} - y h{\left (y \right )} + \frac{1}{3} y + \frac{1}{2} h^{2}{\left (y \right )} - \frac{2}{3} h{\left (y \right )} + \frac{1}{4} dy $$
#
# Now we have to find a function $h$ that is going to minimize this. Solving for a function, as opposed to solving for a number, is generally very, very hard, but because we are integrating over a finite interval, we can use the Euler-Lagrange method from variational calculus to take the derivative of the integnp.random.rand with respect to the function $h(y)$ and set it to zero. Euler-Lagrange methods will be the topic of a later section, but for now we just want the result, namely,
#
# $$ 2 y h{\left (y \right )} - y + h{\left (y \right )} - \frac{2}{3} =0 $$
#
# Solving this gives
#
# $$ h_{opt}(y)= \frac{3 y + 2}{6 y + 3} $$
#
# Finally, we can try solving this using our inner product as
#
# $$ \mathbb{E}( (X - h(Y)) Y ) = 0$$
#
# Writing this out gives,
#
# $$ \int_0^1 \int_0^1  (x-h(y))(x+y) dx dy = \int_0^1 \frac{y\,\left(2 + 3\,y -  3\,\left( 1 + 2\,y \right) \,h(y)\right) }{6} dy = 0$$
#
# and if this is zero everywhere, then the integnp.random.rand must be zero,
#
# $$ 2 y + 3 y^2 - 3 y h(y) - 6 y^2 h(y)=0 $$
#
# and solving this for $h(y)$ gives the same solution:
#
# $$ h_{opt}(y)= \frac{3 y + 2}{6 y + 3} $$
#
# Thus, doing it by the definition, optimization, or inner product gives us the same answer; but, in general, no method is necessarily easier because they both involve potentially difficult or impossible integration, optimization, or functional equation solving. The point is that now that we have a deep toolbox, we can pick and choose which tools we want to apply for different problems.
#
# Before we leave this example, let's use `sympy` to verify the length of the error function we found earlier:
#
# \begin{equation}
# \mathbb{E}(X-  \mathbb{E}(X|Y))^2 = \mathbb{E}(X)^2 - \mathbb{E}(\mathbb{E}(X|Y) )^2
# \end{equation}
#
# that is based on the Pythagorean theorem.

# [1]


from sympy.abc import y, x
from sympy import integrate, simplify

fxy = x + y                 # joint density
fy = integrate(fxy, (x, 0, 1))  # marginal density
fx = integrate(fxy, (y, 0, 1))  # marginal density

h = (3 * y + 2) / (6 * y + 3)  # conditional expectation
LHS = integrate((x - h)**2 * fxy, (x, 0, 1), (y, 0, 1))  # from the definition
RHS = integrate((x)**2 * fx, (x, 0, 1)) - integrate((h)**2 *
                                                    fy, (y, 0, 1))  # using Pythagorean theorem
print(simplify(LHS - RHS) == 0)


# Summary
# --------
#
#  this section, we have pulled together all the projection and least-squares optimization ideas from the previous posts to draw a clean line between our geometric notions of projection from vectors in $\mathbb{R}^n$ to general np.random.random variables. This resulted in the remarkable realization that the conditional expectation is in fact a projection operator for np.random.random variables. The key idea is that because we have these relationships, we can approach difficult problems in multiple ways, depending on which way is more intuitive or tractable in a particular situation. In these pages, we will again and again come back to these intuitions because they form the backbone of statistical signal processing.
#
#
#  the next section, we will have a lot of fun with these ideas working out some examples that are usually solved using more general tools from measure theory.
#
# Note that the book by Mikosch (1998) has some excellent sections covering much of this material in more detail. Mikosch has a very geometric view of the material as well.

# Appendix
# ---------
#
# #### Proof of MMSE of Conditional expectation by Optimization
#
# $$ \min_h \int_\mathbb{R^2} | X - h(Y) |^2 f_{x,y}(X,Y) dx dy $$
#
# $$ \min_h \int_\mathbb{R^2} | X |^2 f_{x,y}(X,Y) dx dy + \int_\mathbb{R^2} | h(Y) |^2 f_{x,y}(X,Y) dx dy - \int_\mathbb{R^2} 2 X h(Y) f_{x,y}(X,Y) dx dy $$
#
# Now, we want to maximize the following:
#
# $$ \max_h \int_\mathbb{R^2}  X h(Y) f_{x,y}(X,Y) dx dy  $$
#
# Breaking up the integral using the definition of conditional expectation
#
# $$ \max_h \int_\mathbb{R}   \left(\int_\mathbb{R} X  f_{x|y}(X|Y) dx \right)h(Y) f_Y(Y)   dy  $$
#
# $$ \max_h \int_\mathbb{R} \mathbb{E}(X|Y) h(Y)f_Y(Y)   dy  $$
#
# From properties of the Cauchy-Schwarz inequality, we know that the maximum happens when $h_{opt}(Y) = \mathbb{E}(X|Y)$, so we have found the optimal $h(Y)$ function as :
#
# $$ h_{opt}(Y) = \mathbb{E}(X|Y)$$
#
# which shows that the optimal function is the conditional expectation.

# ### References
#
# This post was created using the [nbconvert](https://github.com/ipython/nbconvert) utility from the source [IPython Notebook](www.ipython.org) which is available for [download](https://github.com/unpingco/Python-for-Signal-Processing/blob/master/Conditional_Expectation_Projection.ipynb) from the main github [site](https://github.com/unpingco/Python-for-Signal-Processing) for this blog.

# ### Bibliography
#
# * Nelson, Edward. Radically Elementary Probability Theory.(AM-117). Vol. 117. Princeton University Press, 1987.
#
# * Mikosch, Thomas. Elementary stochastic calculus with finance in view. Vol. 6. World Scientific Publishing Company Incorporated, 1998.
#
