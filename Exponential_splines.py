#!/usr/bin/env python
# coding: utf-8

# Start notebook with --profile=sympy flag.
# 
# Following the 1968 paper by Spath. Consider special case with equally spaced abscissae

# [1]


get_ipython().run_line_magic('qtconsole', '')


# [2]


(A, B, C, D)=symbols('A:D') # coefficients
p = symbols('p') # tension parameter
y1p = symbols('y1p') # derivative a left endpoint
ynp = symbols('ynp') # derivative a right endpoint
xcp = symbols('xcp') # x control point
ycp = symbols('ycp') # y control point


# This is the $k^{th}$ piece of the $n$-piece interpolant, 
# 
# $$f_k(x)  = A_k+B_k (x-x_k) + C_k \exp(p_k (x-x_k)) +D_k \exp(-p_k (x-x_k))$$
# 
# given the derivative of the target function at $x(0)$ and at the other end $x(n-1)$.
# 

# [3]


f = A(k)+B(k)*(x-x(k)) + C(k)*exp(p(k)*(x-x(k))) +D(k)*exp(-p(k)*(x-x(k)))


# Sample data to interpolate

# [4]


X =[0,xcp,1]
Y =[0,ycp,1]


# Set up each piece of interpolant

# [5]


c=[f.subs(x(k),X[i]).subs(k,i) for i in range(2)]


# Left-side Interpolation conditions

# [6]


cond_i=[(Y[i]-f.subs(x,x(k)).subs(k,i)) for i in range(2)] # conditions for interpolation
cond_i+= [ Y[2]- c[1].subs(x,X[2])]


# Match end-point 1st derivatives from givens

# [7]


cond_end=[ diff(f,x).subs(k,0).subs(x(0),X[0]).subs(x,X[0]) - y1p,
           diff(f,x).subs(k,1).subs(x(1),X[1]).subs(x,X[2]) - ynp,
 ]
cond_end


# ner continuity conditions

# [8]


cond_cont=[] # continuity conditions
for i,j,v in zip(c[:-2],c[2:],range(1,3)):
    cond_cont.append( (i-j).subs(x,v) )
cond_cont

cond_cont=[(c[0]-c[1]).subs(x,X[1])]


# ner second derivatives must match

# [9]


d2=[i.diff(x,2) for i in c]
cond2_cont=[] # 2nd derivative continuity conditions
for i,j,v in zip(d2[:-2],d2[2:],range(3)):
    cond2_cont.append( (i-j).subs(x,v) )
    
cond2_cont= [diff(c[0]-c[1],x,2).subs(x,X[1])]


# ner first derivatives must match

# [10]


d=[i.diff(x) for i in c]
cond1_cont=[] # 2nd derivative continuity conditions
for i,j,v in zip(d[:-2],d[2:],range(3)):
    cond1_cont.append( (i-j).subs(x,v) )
    
cond1_cont=[diff(c[0]-c[1],x).subs(x,X[1])]


# [11]


len(cond_i)+len(cond_cont)+len(cond_end)+len(cond2_cont)+len(cond1_cont)


# [12]


for i in (cond_i+cond_cont+cond_end+cond2_cont):
    print i.subs(p(k),0)


# [13]


linsys=(cond_i+cond_cont+cond_end+cond2_cont+cond1_cont)
M=Matrix([[ l.coeff(i) for i in flatten([[A(j),B(j),C(j),D(j)] for j in range(2)]) ] for l in linsys])


# [14]


sum([abs(diff(i,x,2)) for i in c]) # curvature metric


# [15]


print c[0]
print c[1]


# [15]




