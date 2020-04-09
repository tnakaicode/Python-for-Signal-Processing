#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
from IPython.display import display, clear_output
import time 


# ## Revisiting Linear Regression 

# In our previous discussion  on linear regression, we saw that regularization  can help trade bias against  variance 

# In[3]:


a = 6;b=1
n=50

x = linspace(0,1,n)
y  = a*x + b +randn(n)
p_,cov_ = polyfit(x,y,1,cov=True)


# In[4]:


def add_cone(ax,p_,cov_,x,alpha=0.95):
    erra_=stats.norm(p_[0],sqrt(cov_[0,0])).interval(alpha)
    errb_=stats.norm(p_[1],sqrt(cov_[1,1])).interval(alpha)
    ax.fill_between(x,polyval([erra_[0],errb_[0]],x),
                      polyval([erra_[1],errb_[1]],x),
                      color='gray',alpha=.3)
    
def add_histogram(ax,x,y):
    p_ = polyfit(x,y,1)
    errs= polyval(p_,x)-y
    ax.hist(errs,alpha=.3,normed=True)
    rvs=stats.norm(mean(errs),std(errs))
    xs = linspace(errs.min(),errs.max(),20)
    ax.plot(xs,rvs.pdf(xs))
    return errs
    
def add_fit(ax,x,y,**kwds):
    p,cov_ = polyfit(x,y,1,cov=True)
    ax.plot(x,polyval(p,x),**kwds)
    return (p,cov_)


# In[5]:


fig,axs=subplots(1,2)
fig.set_size_inches((10,3))
outlier = [0.5,7]

ax=axs[0]
ax.plot(x,y,'o',alpha=.3)
ax.plot(x,polyval(p_,x))
erra_=stats.norm(p_[0],sqrt(cov_[0,0])).interval(.95)
errb_=stats.norm(p_[1],sqrt(cov_[1,1])).interval(.95)
ax.fill_between(x,polyval([erra_[0],errb_[0]],x),
                  polyval([erra_[1],errb_[1]],x),
                  color='gray',alpha=.3)
ax.plot(outlier[0],outlier[1],'o',color='r')

ax = axs[1]
ax.hist(polyval(p_,x)-y,alpha=.3)
ax.vlines(outlier[1]-polyval(p_,outlier[0]),0,ax.get_ylim()[1],color='r',lw=3.)


# ## Try re-fitting with new data point

# In[6]:


# parameter wanders 
fig,axs=subplots(2,2,sharex=True,sharey=True)
fig.set_size_inches((8,4))
a_list = range(6,14,2)

for a,ax in zip(a_list,axs.flat):
    y  = a*x + b +randn(n)
    p_,cov_ = polyfit(x,y,1,cov=True)
    ax.plot(x,y,'o',alpha=.3)
    ax.plot(x,polyval(p_,x))
    ax.set_title("a=%g"%a)
    add_cone(ax,p_,cov_,x)


# In[7]:


a0=6
y0  = a0*x + b +randn(n)
x0 = x[:]
a1 = 15
y1  = a1*x + b +randn(n)
x1 = x[:]

p_,cov_=polyfit(hstack([x,x]),hstack([y0,y1]),1,cov=True)

fig,axs=subplots(1,2)
fig.set_size_inches((8,4))
ax=axs[0]
add_cone(axs[0],p_,cov_,x)
ax.plot(x,polyval([a0,b],x),label='a=6')
ax.plot(x,polyval([a1,b],x),label='a=12')
ax.plot(x,polyval(p_,x),label='all fit')
ax.legend(loc=0)
add_histogram(axs[1],hstack([x,x]),hstack([y0,y1]));


# In[8]:


fig,axs=subplots(1,3)
fig.set_size_inches((10,3))

ax =axs[0]
xc = []
yc = []
outliers=[]

# allow for permutations on wander
# idx=random.permutation(range(len(x)))
# x1 = x1[idx]
# y1 = y1[idx]

# for baseline population 
p_base = polyfit(x0,y0,1)
errs=polyval(p_base,x0)-y0
rvs=stats.norm(mean(errs),std(errs))

# convenience
def eval_likelihood(i,j):
    return rvs.pdf(j-polyval(p_base,i))

for i,j in zip(x1,y1):
    ax.plot(x,y0,'o',alpha=.3)
    if eval_likelihood(i,j) < 0.05:
        outliers.append((i,j))
    else:
        xc.append(i)
        yc.append(j)
    if outliers: 
        outlx = array(outliers)[:,0]
        outly = array(outliers)[:,1]
        ax.plot(outlx,outly,'or',alpha=.6)
        if len(outliers)>5:
            errs=add_histogram(axs[2],outlx,outly)
            axs[2].vlines(j-polyval(polyfit(outlx,outly,1),i),0,0.3,lw=3.,color='r')
            axs[2].set_title('%3.3g'%(stats.kstest(errs,'norm')[1]))
            p_,cov_=add_fit(ax,outlx,outly,lw=3.,color='r')
            add_cone(ax,p_,cov_,allx)

    ax.plot(xc,yc,'o',color='r',alpha=.3)
    allx = x.tolist()+xc
    ally = y0.tolist()+yc
    p_,cov_=add_fit(ax,allx,ally,color='k',lw=2)
    add_cone(ax,p_,cov_,allx)
    errs=add_histogram(axs[1],allx,ally)
    axs[1].set_title('%3.3g'%(stats.kstest(errs,'norm')[1]))
    axs[1].vlines(j-polyval(polyfit(allx,ally,1),i),0,0.3,lw=3.,color='r')
    time.sleep(.1/2)
    clear_output(wait=True)
    display(fig)
    ax.cla()
    ax.axis(ymin=-2,ymax=15,xmin=0,xmax=x0.max())
    axs[1].cla()
    axs[2].cla()

plt.close()


# In[9]:


class AdaptiveRegression(object):
    def __init__(self,x0,y0):
        self.x0,self.y0 = x0,y0
        self.p0, self.cov0 = polyfit(x0,y0,1,cov=True)
        self.errs = y0-polyval(self.p0,x0)
        self.rvs = stats.norm(mean(errs),std(errs))
    def predict(self,i):
        return polyval(self.p0,i)
    def likelihood(self,x,y):
        return self.rvs.pdf(y-self.predict(x))
    def update(self,x,y):
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
            self.x0,self.y0  = hstack([self.x0,y]),hstack([self.y0,y])
        else:
            self.x0,self.y0  = hstack([ self.x0,[x] ]),hstack([ self.y0,[y] ])
        self.p0, self.cov0 = polyfit(self.x0,self.y0,1,cov=True)
        self.errs = self.y0-polyval(self.p0,self.x0)
        self.rvs = stats.norm(mean(errs),std(errs))
    def plot(self,ax,**kwds):
        ax.plot(self.x0,self.y0,marker='o',ls='none',alpha=.5,**kwds)
        ax.plot(self.x0,self.predict(self.x0))
    def add_cone(self,ax,alpha=0.95):
        p_ = self.p0
        cov_ = self.cov0
        x = self.x0
        erra_=stats.norm(p_[0],sqrt(cov_[0,0])).interval(alpha)
        errb_=stats.norm(p_[1],sqrt(cov_[1,1])).interval(alpha)
        ax.fill_between(x,polyval([erra_[0],errb_[0]],x),
                          polyval([erra_[1],errb_[1]],x),
                          color='gray',alpha=.3)
    def add_histogram(self,ax):
        errs =self.errs
        ax.hist(errs,alpha=.3,normed=True)
        xs = linspace(errs.min(),errs.max(),40)
        ax.plot(xs,self.rvs.pdf(xs))


# In[10]:


fig,ax= subplots()
ar=AdaptiveRegression(x0,y0)
ar.plot(ax,color='r')
ar.add_cone(ax)


# In[11]:


fig,axs= subplots(1,2)
fig.set_size_inches((8,3))

ar=AdaptiveRegression(x0,y0)
for i,j in zip(x1,y1):
    ax=axs[0]
    ar.plot(ax,color='b')
    ar.add_cone(ax)
    ax.plot(i,j,'ro')
    ar.update(i,j)
    ar.add_histogram(axs[1])
    axs[1].vlines(j-ar.predict(i),0,0.3,lw=3.,color='r')
    time.sleep(.1/2)
    clear_output(wait=True)
    display(fig)
    ax.cla()
    ax.axis(ymin=-2,ymax=15,xmin=0,xmax=x0.max())
    ax.set_title(repr(ar.p0))
    axs[1].cla()

plt.close()


# In[11]:





# In[11]:




