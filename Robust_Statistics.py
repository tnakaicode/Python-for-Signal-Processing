#!/usr/bin/env python
# coding: utf-8

# [1]


import numpy as np;import matplotlib.pyplot as plt
from IPython.html.widgets import interact
from scipy import  stats
import seaborn as sns
import pandas as pd


# Previously, we talk about maximum likelihood estimation and maximum a-posteriori estimation and in each case we started out with a probability density function of some kind and we further assumed that the samples were identically distributed and independent. The idea behind robust statistics is to construct estimators that can survive the weakening of either or both of these assumptions.
# 
# The first idea to consider is the notion of *location*, which is  a generalization of the idea of "central value". Typically, we just use an estimate of the np.mean for this, but we will see shortly why that is a bad idea.  The general idea of Location satisfies the following requirements
# 
# Let $ X $ be a np.random.random variable with distribution $ F $, and let $\theta(X)$ be some descriptive
# measure of $F$. Then $\theta(X)$ is said to be a measure of *location* if for any constants *a* and *b*, we have the following:
# 
# $$ \theta(X+b) = \theta(X) +b $$
# 
# $$ \theta(-X) = -\theta(X)$$
# 
# $$ X \ge 0 \Rightarrow \theta(X)  \ge 0 $$
# 
# $$ \theta(a X) = a\theta(X) $$
# 
# The first condition is called *location equivariance* (or *shift-invariance*  in signal processing lingo). The fourth condition is called *scale equivariance*, which np.means that the units that $X$ is measured in should not effect the value of the  location estimator.  These Requirements capture the idea of what we intuitively np.mean by *centrality* of a distribution, or where most of the probability mass is located.
# 
# For example, the np.mean estimator is $ \hat{\mu}=\frac{1}{n}\sum X_i $. The first requirement is obviously satisfied as $ \hat{\mu}=\frac{1}{n}\sum (X_i+b) = b +  \frac{1}{n}\sum X_i =b+\hat{\mu}$. Let us consider the second requirement:$ \hat{\mu}=\frac{1}{n}\sum -X_i = -\hat{\mu}$. Finally, the last requirement is satisfied with $ \hat{\mu}=\frac{1}{n}\sum a X_i =a \hat{\mu}$.

# ## What do we np.mean by robust estimators?

# Now that we have the generalized location of centrality embodied in the *location* parameter, what can we do with it?  The next idea is to nail down is the concept of * robust* estimators. Previously, we assumed that our samples were all identically distributed. The key idea is that the samples might be actually coming from a distribution that is contaminated by another nearby distribution, as in the following:
# 
# $$ F(X) = \epsilon G(X) + (1-\epsilon)H(X) $$
# 
# where $ \epsilon $ is between zero and one. This np.means that our data samples $\lbrace X_i \rbrace$ actually derived from two separate distributions, $ G(X) $ and $ H(X) $. We just don't know how they are mixed together. What we really want  is an estimator  that captures the location of $ G(X) $ in the face of np.random.random intermittent contamination by $ H(X) $. It can get even worse than that because we don't know that there is only one contaminating $H(X)$ distribution out there. There may be a whole family of distributions that are contaminating $G(X)$ that we don't know of. This np.means that whatever estimators we construct have to be derived from families of distributions instead of a distribution, which is what we have been assuming for maximum-likelihood  estimators. This is what makes robust estimation so difficult --- the extended theory has to deal with spaces of function distributions instead of particular parameters of a particular probability distribution.
# 
# * Influence function
# * Outlier Detection
# * Estimates of location
#     - definition of location
# * Trimmed np.means
# * Windsorized np.means
# * Hodges Lehmann statistics
# * Asymptotic efficiency
# * Fisher Consistent
# 
# * Robust Regression 
#     
# 
#     - least median
#     - outliers

# [2]


n0=stats.norm(0,1)
n1=stats.norm(0,10)
xi = np.linspace(-5,5,100)

fig,ax=plt.subplots()
ax.plot(xi,n0.pdf(xi))
ax.plot(xi,n1.pdf(xi))

def bias_coin(phead = .5):
    while True:
        yield int( np.np.random.random.np.random.rand() < phead ) 

pct_mixed  = 0.1
bias_coin_gen = bias_coin(pct_mixed)        
dual_set = [n0,n1]
samples  = [ dual_set[bias_coin_gen.next()].rvs() for i in range(500) ]


# [3]


hist(samples,bins=20)
title('average = %3.3f, median=%3.3f pct_mixed=%3.3f'%(np.mean(samples),np.median(samples),pct_mixed))


# [4]


import sympy.stats
from sympy.abc import x
eps = sympy.symbols('epsilon')

mixed_cdf = sympy.stats.cdf(sympy.stats.Normal('x',0,1),'x')(x)*(1-eps) + eps*sympy.stats.cdf(sympy.stats.Normal('x',1,2),'x')(x)
mixed_pdf = sympy.diff(mixed_cdf,x)


# [5]


def plot_mixed_dist(epsilon=.1):
    n1 = stats.norm(1,2)
    xi = np.linspace(-5,5,100)
    fig,ax = plt.subplots()
    ax.plot(xi,[sympy.lambdify(x,mixed_pdf.subs(eps,epsilon))(i) for i in xi],label='mixed',lw=2)
    ax.plot(xi,n0.pdf(xi),label='g(x)',linestyle='--')
    ax.plot(xi,n1.pdf(xi),label='h(x)',linestyle='--')
    ax.legend(loc=0)
    ax.set_title('epsilon = %2.2f'%(epsilon))
    ax.vlines(0,0,.4,linestyle='-',color='g')
    ax.vlines(epsilon,0,.4,linestyle='-',color='b')

interact(plot_mixed_dist,epsilon=(0,1,.05))


# ## M-estimators

# M-estimators are generalized maximum likelihood estimators. Recall that for maximum likelihood, we want to maximize the likelihood function as in the following:
# 
# $$ L_{\mu}(x_i) = \prod f_0(x_i-\mu)$$
# 
# and then to find the estimator $\hat{\mu}$ so that
# 
# $$ \hat{\mu} = \arg \max_{\mu} L_{\mu}(x_i) $$
# 
# So far, everything is the same as our usual maximum-likelihood  derivation except for the fact that we don't know $f_0$, the distribution of the $\lbrace X_i\rbrace$. Making the convenient definition of
# 
# $$ \rho = -\log f_0 $$
# 
# we obtain the more convenient form of the likelihood product and the optimal $\hat{\mu}$ as
# 
# $$ \hat{\mu} = \arg \min_{\mu} \sum \rho(x_i-\mu)$$
# 
# If $\rho$ is differentiable, then differentiating  this with respect to $\mu$ gives
# 
# $$ \sum \psi(x_i-\hat{\mu}) = 0 $$
# 
# with $\psi = \rho'$ and for technical reasons we will assume that $\psi$ is increasing. The key idea here is we want to consider general $\rho$ functions that my not be MLE for *any* distribution.
# 

# ### The distribution of  M-estimates 

# For a given distribution $F$, we define $\mu_0=\mu(F)$ as the solution to the following 
# 
# $$ \mathbb{E}_F(\psi(x-\mu_0))= 0 $$
# 
# It is technical to show, but it turns out that $\hat{\mu} \sim \mathcal{N}(\mu_0,\frac{v}{n})$ with
# 
# $$ v = \frac{\mathbb{E}_F(\psi(x-\mu_0)^2)}{(\mathbb{E}_F(\psi^\prime(x-\mu_0)))^2} $$
# 
# Thus, we can say that $\hat{\mu}$ is asymptotically normal with asymptotic value $\mu_0$ and asymptotic variance $v$. This leads to the efficiency ratio which is defined as  the following:
# 
# $$ \texttt{Eff}(\hat{\mu})= \frac{v_0}{v} $$
# 
# where $v_0$ is the asymptotic variance of the MLE and measures how near $\hat{\mu}$ is to the optimum. for example, if for two estimates with asymptotic variances $v_1$ and $v_2$, we have $v_1=3v_2$, then first estimate requires three times as many observations to obtain the same variance as the second.
# 
# For example, for the sample np.mean (i.e. $\hat{\mu}=\frac{1}{n} \sum X_i$) with $F=\mathcal{N}$, we have $\rho=x^2/2$ and $\psi=x$ and also $\psi'=1$. Thus, we have $v=\mathbb{V}(x)$. Alternatively, using the sample median as the estimator for the location, we have $v=\frac{1}{4 f(\mu_0)^2}$. Thus, if we have $F=\mathcal{N}(0,1)$, for the sample median, we obtain $v=\frac{2\pi}{4} \approx 1.571$. This np.means that the sample median takes approximately 1.6 times as many samples to obtain the same variance for the location as the sample np.mean.

# One way to think about M-estimates is a weighted np.means. Most of the time, we have $\psi(0)=0$ and $\psi'(0)$ exists so that $\psi$ is approximately linear at the origin. Using the following definition:
# 
# 
# $$ W(x)  =  \begin{cases}
#                 \psi(x)/x & \text{if} \: x \neq 0 \\
#                 \psi'(x)  & \text{if} \: x =0 
#             \end{cases}
# $$
# 
# We can write our earlier equation as follows:
# 
# $$ \sum W(x_i-\hat{\mu})(x_i-\hat{\mu}) = 0 $$
# 
# Solving this for $\hat{\mu} $ yields the following,
# 
# $$ \hat{\mu} = \frac{\sum w_{i} x_i}{\sum w_{i}} $$
# 
# where $w_{i}=W(x_i-\hat{\mu})$. The question that remains is how to pick the $\psi$ functions.

# ### Huber functions

# The family of Huber function is defined by the following:
# 
# $$ \rho_k(x ) = \begin{cases}
#                 x^2  & \text{if} \: |x|\le k \\
#                 2 k |x|-k^2 & \text{if} \: |x| \gt k
#                 \end{cases}
# $$
# 
# with corresponding derivatives $2\psi_k(x)$ with
# 
# $$ \psi_k(x ) = \begin{cases}
#                 x  & \text{if} \: |x|\le k \\
#                 \text{sgn}(x)k & \text{if} \: |x| \gt k
#                 \end{cases}
# $$
# where the limiting cases $k \rightarrow \infty$ and $k \rightarrow 0$ correspond to the np.mean and median, respectively. To see this, take $\psi_{\infty} = x$ and therefore $W(x) = 1$ and thus the defining equation results in
# 
# $$ \sum_{i=1}^{n} (x_i-\hat{\mu}) = 0 $$
# 
# and then solving this leads to $\hat{\mu} = \frac{1}{n}\sum x_i$. Note that choosing $k=0$ leads to  the sample median, but that is not so straightforward to solve for.

# [6]


fig,ax=plt.subplots()
colors=['b','r']
for k in [1,2]
    ax.plot(xi,np.ma.masked_np.array(xi,abs(xi)>k),color=colors[k-1])
    ax.plot(xi,np.ma.masked_np.array(np.sign(xi)*k,abs(xi)<k),color=colors[k-1],label='k=%d'%k)
ax.axis(ymax=2.3,ymin=-2.3)
ax.set_ylabel(r'$\psi(x)$',fontsize=28)
ax.set_xlabel(r'$x$',fontsize=24)
ax.legend(loc='best')
ax.set_title('Huber functions')
ax.grid()


# The $W$ function corresponding to Huber's $\psi$ is the following:
# 
# $$ W_k(x) = \min\Big{\lbrace} 1, \frac{k}{|x|} \Big{\rbrace} $$
# 
# which is plotted in the following cell for a few values of $k$.

# [7]


fig,ax=plt.subplots()
ax.plot(xi,np.vstack([np.ones(xi.shape),2/abs(xi)]).min(axis=0),label='k=2')
ax.plot(xi,np.vstack([np.ones(xi.shape),1/abs(xi)]).min(axis=0),label='k=1')
ax.axis(ymax=1.1)
ax.legend(loc=0)
ax.set_title("Huber's weight function")


# Another alternative intuitive way  to interpret  the M-estimate is to rewrite the following:
# 
# $$ \hat{\mu} = \hat{\mu} +\frac{1}{n}\sum_i \psi(x_i-\hat{\mu}) = \frac{1}{n} \sum_i \zeta(x_i,\hat{\mu})$$
# 
# which for the Huber family of functions takes on the form :
# 
# $$ \zeta(x,\mu) = \begin{cases}
#                     \mu - k & \text{if} \: x \lt \mu-k \\
#                     x & \text{if} \: \mu-k \le x \le \mu+k \\
#                     \mu+k & \text{if} \: x \gt \mu \\
#                   \end{cases}
# $$
# 
# Thus, the interpretation here is that $\hat{\mu}$ is the average of the truncated pseudo-observations $\zeta_i$ where the observations beyond a certain point are clipped at the $k$-offset of the $\hat{\mu}$. 

# ## Asymptotic variance

# [8]


from sympy import mpmath, symbols, diff, Piecewise, sign, lambdify
from sympy.stats import density, cdf, Normal
from sympy.abc import k,x


# [9]


eps = symbols('epsilon')
lpdf=diff(cdf(Normal('x',0,1))(x)*(1-eps)+ eps*cdf(Normal('x',0,10))(x),x)
p = Piecewise((x,abs(x)<k),(k*sign(x),True))


# [10]


def closure_on_asymptotic_variance(mn=(0,0),std=(1,10)):
    from sympy.abc import k,x
    eps = symbols('epsilon')
    lpdf=diff(cdf(Normal('x',mn[0],std[0]))(x)*(1-eps)+ eps*cdf(Normal('x',mn[1],std[1]))(x),x)
    def asymptotic_variance(kval,epsval):
        p = Piecewise((x,abs(x)<kval),(kval*sign(x),True))
        denom=mpmath.quad(lambdify(x,lpdf.subs(eps,epsval)),[-kval,kval])**2
        numer=mpmath.quad(lambdify(x,(p.subs(k,kval))**2*lpdf.subs(eps,epsval)),[-np.inf,np.inf])
        return float(numer/denom)
    return asymptotic_variance


# [11]


case2=closure_on_asymptotic_variance()


# [12]


kvals= [.001,0.3,0.5,.7,1.00001,1.4,1.7,2,3,4,5]


# [13]


fig2,ax=plt.subplots()
ax.plot(kvals,[case2(k,0) for k in kvals],'-o',label='eps=0')
ax.plot(kvals,[case2(k,.05) for k in kvals],'-o',label='eps=.05')
ax.plot(kvals,[case2(k,.1) for k in kvals],'-o',label='eps=0.1')
ax.set_xlabel("k")
ax.set_ylabel("asymptotic variance")
ax.legend(loc=0)
ax.set_title(r"$\mathcal{N}(0,1) , \mathcal{N}(0,10)$ mixed",fontsize=18)


# ## Computable Example

# [14]


nsamples = 500
ncols = 300
xs = np.np.array([dual_set[bias_coin_gen.next()].rvs() for i in range(ncols)*nsamples ]).reshape(nsamples,-1)


# [15]


fig,ax=plt.subplots()
ax.hist(np.np.mean(xs,0),20,alpha=0.8,label = 'np.mean')
ax.hist(np.median(xs,0),20,alpha=0.3,label ='median')
ax.legend()


# [16]


fig,ax=plt.subplots()
sns.violinplot(np.vstack([np.median(xs,axis=0),np.np.mean(xs,axis=0)]).T,ax=ax,names=['median','np.mean']);


# [17]


def huber_estim(x,k=1,tol=1e-6):
    if x.ndim==2: # loop over columns
        out = []        
        for i in range(x.shape[1]):
            out.append(huber_estim(x[:,i],k=k)) # recurse        
        return np.np.array(out)
    else:
        mu = median(x)
        mad = median(abs(x-mu))*1.4826 # follow MADN convention
        while True:
            mu_i=np.mean(minimum(maximum(mu-k*mad,x),mu+k*mad))
            if abs(mu-mu_i) < tol*mad: break
            mu = mu_i
        return mu_i


# [18]


huber_est={k:huber_estim(xs,k) for k in [1,1.5,2,3]}
huber_est[0] = np.median(xs,axis=0)
huber_est[4] = np.np.mean(xs,axis=0)
fig,ax=plt.subplots()
sns.violinplot(pd.DataFrame(huber_est),ax=ax)
ax.set_xticklabels(['median','1','1.5','2','3','np.mean']);


# [19]


huber_est={k:huber_estim(xs,k) for k in kvals}
huber_est_df = pd.DataFrame(huber_est)


# [20]


fig,ax=plt.subplots()
sns.violinplot(huber_est_df,ax=ax)


# [21]


fig2,ax=plt.subplots()
ax.plot(kvals,[case2(k,pct_mixed) for k in kvals],'-o',label='eps=0.1')
(huber_est_df.var()*nsamples).plot(marker='o',label='est eps=0.1',ax=ax)
ax.set_xlabel("k")
ax.set_ylabel("asymptotic variance")
ax.legend(loc=0)


# ## References

# * Maronna, R. A., R. D. Martin, and V. J. Yohai. "Robust Statistics: Theory and Methods". 2006.
