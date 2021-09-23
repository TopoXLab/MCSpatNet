import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot


import pandas as pd
import numpy as np
import pyper as pr
import rpy2.robjects as robj
from rpy2.robjects import pandas2ri
from rpy2.robjects import FloatVector as rfc
from rpy2.robjects import StrVector as rsc
from rpy2.robjects import FactorVector as rfctc
from rpy2.robjects import ListVector as rlist
from rpy2.robjects.packages import importr
import rpy2.rinterface as ri
import matplotlib.pyplot as plt

# uncomment to install
# !pip install pyper
# !pip install rpy2
# r("install.packages('spatstat', repos = \"https://cloud.r-project.org\")")


# In[310]:


pandas2ri.activate()
base = importr('base')
spatstat = importr('spatstat')
rlist = ri.baseenv['list']
rnull = ri.NULL
rcmd = pr.R(use_numpy=True)

def ppp(x, y, marks, window_xrange=None, window_yrange=None):
    '''
    create a point pattern object
    
    input:
        x: list of the input points' x coordinate
        y: list of the input points' y coordinate
        marks: list of the mark of input points
        window_xrange: a list that specifies the x range of the window. Equals [min(x), max(x)] if it is None
        window_yrange: a list that specifies the y range of the window. Equals [min(y), max(y)] if it is None
        
    output:
        point pattern dictionary
    
    >>> x = [1, 2, 3, 4]
    >>> y = [1, 2, 3, 4]
    >>> marks = ['a', 'a', 'b', 'c']
    >>> pp = ppp(x, y, marks, window_xrange=[0, 5], window_yrange=[0, 5])
    >>> r.print(pp)
    '''
    pp = {}
    if not window_xrange:
        window_xrange=rfc([robj.r.min(rfc(x)), robj.r.max(rfc(x))])
    if not window_yrange:
        window_yrange=rfc([robj.r.min(rfc(y)), robj.r.max(rfc(y))])
        
    pp['ppp'] = spatstat.ppp(x=rfc(x), y=rfc(y), 
                             window=spatstat.owin(xrange=window_xrange, 
                                                  yrange=window_yrange), 
                             marks=rfctc(marks))
    pp['coor'] = [x] + [y]
    pp['marks'] = marks
    return pp

def Gest(pp, r=None, correction='km', plot=True):
    '''
    Take the output of ppp() function as the input and return the G function value at given r 
    
    input:
        pp: point pattern 
        r : your selected radius ticks 
        correction: the border correction you want to apply. 
                    options are 'none', 'rs', 'km' and 'han'
                    the default is 'km'
        plot: if True given the plots
    
    output:
        Gest: the R Gest class object
    
    >>> x = [1, 2, 3, 4]
    >>> y = [1, 2, 3, 4]
    >>> marks = ['a', 'a', 'b', 'c']
    >>> pp = ppp(x, y, marks)
    >>> Gest(pp, plot=True)
    '''
    r_ppp = pp['ppp']
    if r:
        r_Gest = spatstat.Gest(r_ppp, r=rfc(r), correction=correction)
    else:
        r_Gest = spatstat.Gest(r_ppp, correction=correction)
    
    r_used = r_Gest['r']
    G_val_theo = r_Gest['theo']
    G_val_samp = r_Gest['raw' if correction=='none' else correction]
        
    if plot:
        plt.plot(r_used, G_val_samp, c='black', linestyle='-', linewidth=3,  label=r"$G_{"+correction+"}$")
        plt.plot(r_used, G_val_theo, c='red', linestyle='--', linewidth=3, label=r"$G_{Poisson}$")
        plt.xlabel('r')
        plt.ylabel('G(r)')
        plt.legend(fontsize=12)
        plt.title("G function")
        
    return r_Gest

def Kest(pp, r=None, correction='iso', plot=True):
    '''
    Take the output of ppp() function as the input and return the K function value at given r 
    
    input:
        pp: point pattern 
        r : your selected radius ticks 
        correction: the border correction you want to apply. 
                    options are 'border', 'iso' and 'trans'
                    the default is 'isotropic' (which is the Ripley correction)
        plot: if True given the plots
    
    output:
        r_Kest: the R Kest class 
    
    >>> x = [1, 2, 3, 4]
    >>> y = [1, 2, 3, 4]
    >>> marks = ['a', 'a', 'b', 'c']
    >>> pp = ppp(x, y, marks)
    >>> Kest(pp, plot=True)
    '''
    r_ppp = pp['ppp']
    if r:
        r_Kest = spatstat.Kest(r_ppp, r=rfc(r), correction=correction)
    else:
        r_Kest = spatstat.Kest(r_ppp, correction=correction)
    
    r_used = r_Kest['r']
    K_val_theo = r_Kest['theo']
    K_val_samp = r_Kest['un' if correction=='none' else correction]
        
    if plot:
        plt.plot(r_used, K_val_samp, c='black', linestyle='-', linewidth=3,  label=r"$K_{"+correction+"}$")
        plt.plot(r_used, K_val_theo, c='red', linestyle='--', linewidth=3, label=r"$K_{Poisson}$")
        plt.xlabel('r')
        plt.ylabel('K(r)')
        plt.legend(fontsize=12)
        plt.title("K function")
        
    return r_Kest, K_val_samp

def Gcross(pp, i, j, r=None, correction='km', plot=True):
    '''
    Take the output of ppp() function as the input and return the cross G function value at given r 
    
    input:
        pp: point pattern 
        i : the type of point in the center
        j : the type of point in the neighbor
        r : your selected radius ticks 
        correction: the border correction you want to apply. 
                    options are 'none', 'rs', 'km' and 'han'
                    the default is 'km'
        plot: if True given the plots
    
    output:
        r_Gcross: the R Gcross class object 
    
    >>> x = [1, 2, 3, 4]
    >>> y = [1, 2, 3, 4]
    >>> marks = ['a', 'a', 'b', 'c']
    >>> pp = ppp(x, y, marks)
    >>> Gcross(pp, i='a', j='b',  plot=True)
    '''
    r_ppp = pp['ppp']
    if r:
        r_Gcross = spatstat.Gcross(r_ppp, r=rfc(r), i=i, j=j, correction=correction)
    else:
        r_Gcross = spatstat.Gcross(r_ppp, i=i, j=j, correction=correction)
    
    r_used = r_Gcross['r']
    G_val_theo = r_Gcross['theo']
    G_val_samp = r_Gcross['raw' if correction=='none' else correction]
        
    if plot:
        plt.plot(r_used, G_val_samp, c='black', linestyle='-', linewidth=3,  label=r"$G_{"+correction+"}$")
        plt.plot(r_used, G_val_theo, c='red', linestyle='--', linewidth=3, label=r"$G_{Poisson}$")
        plt.xlabel('r')
        plt.ylabel('G(r)')
        plt.legend(fontsize=12)
        plt.title(r"$G_{" + str(i) +  ", " + str(j) + "} (r)$ function")
    
    return r_Gcross

def Kcross(pp, i, j, r=None, correction='iso', plot=True):
    '''
    Take the output of ppp() function as the input and return the cross K function value at given r 
    
    input:
        pp: point pattern 
        i : the type of point in the center
        j : the type of point in the neighbor
        r : your selected radius ticks 
        correction: the border correction you want to apply. 
                    options are 'none', 'iso', 'border', 'board.modif', "trans"
                    the default is 'iso'
        plot: if True given the plots
    
    output:
        r_Kcross: the R Kcross class object
    
    >>> x = [1, 2, 3, 4]
    >>> y = [1, 2, 3, 4]
    >>> marks = ['a', 'a', 'b', 'c']
    >>> pp = ppp(x, y, marks)
    >>> Kcross(pp, i='a', j='b',  plot=True)
    '''
    r_ppp = pp['ppp']
    if r:
        r_Kcross = spatstat.Kcross(r_ppp, r=rfc(r), i=i, j=j, correction=correction)
    else:
        r_Kcross = spatstat.Kcross(r_ppp, i=i, j=j, correction=correction)
    
    r_used = r_Kcross['r']
    K_val_theo = r_Kcross['theo']
    K_val_samp = r_Kcross['un' if correction=='none' else correction]
        
    if plot:
        plt.plot(r_used, K_val_samp, c='black', linestyle='-', linewidth=3,  label=r"$K_{"+correction+"}$")
        plt.plot(r_used, K_val_theo, c='red', linestyle='--', linewidth=3, label=r"$K_{Poisson}$")
        plt.xlabel('r')
        plt.ylabel('K(r)')
        plt.legend(fontsize=12)
        plt.title(r"$K_{" + str(i) +  ", " + str(j) + "} (r)$ function")
        
    return r_Kcross,K_val_samp

def envelop(pp, fun='Kest', i=None, j=None, r=None, correction=None, global_var=False, theoretic_var=False, plot=True):
    '''
    Return Monte Carlo simulated envelop for your spatial statistics
    
    Input:
        pp:  point pattern
        fun: spatial statistics you want envelop. 
             options are: 'Kest', 'Kcross', 'Gest', 'Gcross'
        i  : the type of cell in the center if you use cross statistics
        j  : the type of cell in the neighbor if you use cross statistics
        r  : your selected radius ticks 
        correction: boarder correction you want to apply for your statistics 
        global_var : use the global envelope for your confidence band
        theoretic_var : use theoretic envelope for your confidence band 
        plot : the plot is shown if True
        
    output:
        r_envelop: the R envelope object 
        
    >>> x = [1, 2, 3, 4]
    >>> y = [1, 2, 3, 4]
    >>> marks = ['a', 'a', 'b', 'c']
    >>> pp = ppp(x, y, marks)
    >>> envelop(pp, fun='Kcross', i='a', j='b', correction='iso', plot=True)
    '''
    print('fun',fun)
    r_ppp = pp['ppp']

    if not i:
        i = rnull
    if not j:
        j = rnull
    if not correction:
        correction = rnull
    
    if not r:
        r_envelop = spatstat.envelope(r_ppp, fun=fun, i=i, j=j, correction=correction,fix_marks=True)
    else:
        r_envelop = spatstat.envelope(r_ppp, r=rfc(r), fun=fun, i=i, j=j, correction=correction,fix_marks=True)
    r_used = r_envelop['r']
    enve_val_theo = r_envelop['theo']
    enve_val_samp = r_envelop['obs']
    theo_hi = r_envelop['hi']
    theo_lo = r_envelop['lo']

    if plot:
        plt.plot(r_used, enve_val_samp, c='black', linestyle='-', linewidth=3,  label=r"$"+fun+"_{obs}$")
        plt.plot(r_used, enve_val_theo, c='red', linestyle='--', linewidth=3, label=r"$"+fun+"_{Poisson}$")
        plt.plot(r_used, theo_hi, c='grey', linestyle='-', linewidth=1, label=r"$"+fun+"_{confidence}$")
        plt.plot(r_used, theo_lo, c='grey', linestyle='-', linewidth=1)
        plt.fill_between(r_used, theo_hi, theo_lo, color='grey', alpha=0.4)
        plt.xlabel('r')
        plt.ylabel(fun+'(r)')
        plt.legend(fontsize=12)
        if i or j:
            plt.title(r"$" + fun+ "_{" + str(i) +  ", " + str(j) + "} (r)$ function")
        else:
            plt.title(r"$" + fun+ "(r)$ function")

    return r_envelop

def pool(pp_list, fun=Kest, r=None, i=None, j=None, correction='none'):
    '''
    Pool multiple spatial statistics
    
    Input: 
        pp_list:  python list that contains point patterns
        fn: spatial statistics function to be applied.
        r:  radius ticks where spatial statistics is evaluated at
        i: center points type
        j: neighbor points type
        correction: border correction to be applied
        
    output:
        r : radius tick where G_pool is evaluated at
        G_pool : pooled G function value
        VG_pool : pooled G function's variance
        lamb_pool : pooled intensity value
        
    >>> x1 = [1, 2, 3, 4]; y1 = [1, 2, 3, 4]
    >>> x2 = [2, 2, 3, 4]; y2 = [2, 2, 3, 4]
    >>> x3 = [3, 2, 3, 4]; y3 = [3, 2, 3, 4]
    >>> marks = ['a', 'a', 'b', 'c']
    >>> pp1 = ppp(x1, y1, marks); pp2 = ppp(x2, y2, marks); pp3 = ppp(x3, y3, marks)
    >>> K1 = Kest(pp1); K2 = Kest(pp2); K3 = Kest
    >>> r_pool = pool([K1, K2, K3])
    
    # Or you could pool the envelop also
    >>> Kenv1 = envelop(pp1); Kenv2 = envelop(pp2);  Kenv3 = envelop(pp3);
    >>> G_pool, VG_pool, lambda_pool = pool([Kenv1, Kenv2, Kenv3])
    '''
    
    m = []
    lambda_list = []
    fv_list = []
    W_list = []
    
    for k in range(len(pp_list)):
        ppp_obj = pp_list[k]['ppp']
        marks = pp_list[k]['marks']
        if i or j:
            ni = sum([mark==i for mark in marks])
            nj = sum([mark==j for mark in marks])
        else:
            ni = len(marks)
            nj = ni
        m.append(ni*nj)
        W = ppp_obj[0]
        xrange = W[1][1]-W[1][0]
        yrange = W[2][1]-W[2][0]
        W_size = xrange*yrange
        lambda_list.append(nj/W_size)
        W_list.append(W_size)
        
        if not r:
            max_r = np.sqrt(xrange**2 + yrange**2)/4
            r = [i*(max_r/200) for i in range(200)]
        if i or j:
            fv_list.append(fun(pp_list[k], r=r, i=i, j=j, correction=correction, plot=False))
        else:
            fv_list.append(fun(pp_list[k], r=r, correction=correction, plot=False))
    
    G_pool = [0 for i in range(len(r))]
    VG_pool = [0 for i in range(len(r))]
    for k in range(len(r)):
        Gs = [fv.iloc[k, 2] for fv in fv_list]
        G_pool[k] = sum(np.array(m)*np.array(Gs))/sum(m)
        m_star = len(m)*np.array(m)/sum(m)
        G_star = np.array([x if x>0 else 0 for x in len(fv_list)*np.array(Gs)/sum(Gs)])
        cov_m_G = np.cov(m_star, G_star)
        VG_pool[k] = G_pool[k]**2/len(m)*(cov_m_G[0,0]+cov_m_G[1,1] - 2*cov_m_G[1,0])
    lamb_pool = sum(np.array(W_list)*np.array(lambda_list))/sum(W_list)
    
    return r, G_pool, VG_pool, lamb_pool


# In[278]:
if __name__=="__main__":

    # Read and parse your data
    points = np.load("TCGA-A7-A4SC-01Z-00-DX1_15500_35001_1001_1000_0.9920000000000001_gt_dots.npy", allow_pickle=True)
    cell_code = {1:'lymphocyte', 2:'tumor', 3:'other'}
    x = []
    y = []
    mark = []
    heights, widths, channels = points.shape
    for c in range(channels):
        for h in range(heights):
            for w in range(widths):
                if points[h, w, c]:
                    x.append(h)
                    y.append(w)
                    mark.append(cell_code[c])


    # In[279]:


    test_ppp = ppp(x, y, mark)
    print(test_ppp['ppp'])    # the point pattern in R object
    # print(test_ppp['coor'])   # your coordinate
    # print(test_ppp['marks'])  # your cell types


    # In[280]:


    r_ppp = test_ppp['ppp']
    r = [0, 5, 10, 15, 20, 25]
    r_Gest = spatstat.Gest(r_ppp, r=rfc(r))

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    r_Gest = Gest(test_ppp, correction='none')
    plt.subplot(2, 2, 2)
    r_Kest,K_val_samp = Kest(test_ppp, correction='none')
    plt.subplot(2, 2, 3)
    r_Gcross = Gcross(test_ppp, i='lymphocyte', j='tumor', correction='km')
    plt.subplot(2, 2, 4)
    r_Kcross,K_val_samp = Kcross(test_ppp, i='lymphocyte', j='tumor', correction='iso')


    # Each of these R object is a Pandas dataframe in Python, you could access the content with their keys.For example:

    # In[281]:


    r_Gest.head()
    # r is the radius ticks where G function is evaluated
    # theo is the Poisson process's G value
    # raw is the sample estimation 


    # In[282]:


    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    r_envelop = envelop(test_ppp, fun='Gcross', i='lymphocyte', j='tumor')
    plt.subplot(2, 2, 2)
    r_envelop = envelop(test_ppp, fun='Kcross', i='lymphocyte', j='tumor')
    plt.subplot(2, 2, 3)
    r_envelop = envelop(test_ppp, fun='Gest')
    plt.subplot(2, 2, 4)
    r_envelop = envelop(test_ppp, fun='Kest')


    # ### Statistics Pooling
    # 
    # Statistics pooling is based on following formula (same procedure applies to K function): 
    # 
    # __Process Steps__: <br>
    # Suppose now we have $G_{i,j}(r)^{(1)}, G_{i,j}(r)^{(2)}, \cdots, G_{i,j}(r)^{(Q)}$. And we also have estimation of the intensity $\hat{\lambda}_j^{(1)}, \hat{\lambda}_j^{(2)}, \cdots, \hat{\lambda}_j^{(Q)}$ Then we pool these Q statistics functions to get our final pooled G function. We also pooled these Q intensity estimator to get a pooled intensity and then the pooled theoretic closed-formed G function.
    # 
    # __Pooled $G_{i,j}(r)$ (Theoretic one)__: <br>
    # $\lambda_{pool} = \frac{\sum_{i=1}^{Q} |W_i|\lambda_i}{\sum_{i=1}^{Q}|W_i|}$
    # 
    # $G(r)_{pool} = 1-\exp\{-\lambda_{pool}\pi r^2\}$
    # 
    # __Pooled $\hat{G}_{i,j}(r)$ (Sample version)__: <br>
    # $m_k = n1_k \times n2_k$<br>
    # 
    # $\hat{G}_{i,j}(r) = \frac{\sum_{k=1}^{Q}m_kG_{i,j}^{(k)}(r)}{\sum_{k=1}^{Q}m_k}$
    # 
    # where $n1_k$ is the number of points of the center type, and $n2_k$ is the number of points of the neighbor type. To estimate $G_{i,j}^{(k)}(r)$ we need $n1_k \times n2_k$ pair of samples. 
    # 
    # __Variance for Pooled $\hat{G}_{i,j}(r)$__: <br>
    # $\vec{m}^* = \left(\frac{Q m_1}{\sum_{k=1}^{Q}m_k}, \frac{Q m_2}{\sum_{k=1}^{Q} m_k}, 
    # \cdots, \frac{Q m_k}{\sum_{k=1}^{Q}m_k}\right)$ <br>
    # $\vec{G^*_{i,j}(r)} = \left(\frac{Q G^{(1)}_{i,j}(r)}{\sum_{k=1}^{Q}G^{k}_{i,j}(r)}, 
    #                             \frac{Q G^{(2)}_{i,j}(r)}{\sum_{k=1}^{Q}G^{k}_{i,j}(r)}, 
    #                             \cdots, 
    #                             \frac{Q G^{(k)}_{i,j}(r)}{\sum_{k=1}^{Q}G^{k}_{i,j}(r)}\right)$ <br>
    # $\Sigma = cov\left(\vec{m}^*, \vec{G^*_{i,j}(r)}\right)$ <br>
    # $Var\left(\hat{G}^*_{i,j}(r)\right) = \frac{\hat{G}^*_{i,j}(r)^2}{Q}(\Sigma_{1,1}+\Sigma_{2,2}-2\Sigma_{1,2})$
    # 
    # Where the confidence band width equals $\sqrt{Var\left(\hat{G}_{i,j}(r)\right)}$

    # In[309]:


    # Suppose now you have other two point processes
    x2 = [coor + 150*np.random.rand(1)[0] for coor in x]
    y2 = [coor + 150*np.random.rand(1)[0] for coor in y]
    x3 = [coor + 150*np.random.rand(1)[0] for coor in x]
    y3 = [coor + 150*np.random.rand(1)[0] for coor in y]
    test_pp2 = ppp(x2, y2, marks=np.random.permutation(mark))
    test_pp3 = ppp(x3, y3, marks=np.random.permutation(mark))

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    r_used, G_pool, VG_pool, lamb_pool = pool([test_ppp, test_pp2, test_pp3], fun=Gest)
    # theoretic formula for G function is 1-exp(-\lambda * \pi * r^2)
    G_theo = 1-np.exp(-np.pi*lamb_pool*np.array(r_used)**2)
    plt.plot(r_used, G_pool, c='black', linewidth=3, linestyle='-', label=r"$G_{pooled}(r)$")
    plt.plot(r_used, G_theo, c='red', linewidth=3, linestyle='--', label=r"$G_{Poisson}(r)$")
    plt.plot(r_used, G_pool-np.sqrt(VG_pool), c='grey', linewidth=1, linestyle='-', label="G(r) Confidence Band")
    plt.plot(r_used, G_pool+np.sqrt(VG_pool), c='grey', linewidth=1, linestyle='-')
    plt.fill_between(r_used, G_pool-np.sqrt(VG_pool), G_pool+np.sqrt(VG_pool), color='grey', alpha=0.4)
    plt.legend()
    plt.title("Pooled G Function")

    plt.subplot(2, 2, 2)
    r_used, K_pool, VK_pool, lamb_pool = pool([test_ppp, test_pp2, test_pp3], fun=Kest)
    # theoretic formula for K function is \pi*r^2
    K_theo = np.pi*np.array(r_used)**2
    plt.plot(r_used, K_pool, c='black', linewidth=3, linestyle='-', label=r"$K_{pooled}(r)$")
    plt.plot(r_used, K_theo, c='red', linewidth=3, linestyle='--', label=r"$K_{Poisson}(r)$")
    plt.plot(r_used, K_pool-np.sqrt(VK_pool), c='grey', linewidth=1, linestyle='-', label="K(r) Confidence Band")
    plt.plot(r_used, K_pool+np.sqrt(VK_pool), c='grey', linewidth=1, linestyle='-')
    plt.fill_between(r_used, K_pool-np.sqrt(VK_pool), K_pool+np.sqrt(VK_pool), color='grey', alpha=0.4)
    plt.legend()
    plt.title("Pooled K Function")

    plt.subplot(2, 2, 3)
    r_used, G_pool, VG_pool, lamb_pool = pool([test_ppp, test_pp2, test_pp3], fun=Gcross, i='lymphocyte', j='tumor')
    # theoretic formula for G function is 1-exp(-\lambda * \pi * r^2)
    G_theo = 1-np.exp(-np.pi*lamb_pool*np.array(r_used)**2)
    plt.plot(r_used, G_pool, c='black', linewidth=3, linestyle='-', label=r"$G_{pooled}(r)$")
    plt.plot(r_used, G_theo, c='red', linewidth=3, linestyle='--', label=r"$G_{Poisson}(r)$")
    plt.plot(r_used, G_pool-np.sqrt(VG_pool), c='grey', linewidth=1, linestyle='-', label="G(r) Confidence Band")
    plt.plot(r_used, G_pool+np.sqrt(VG_pool), c='grey', linewidth=1, linestyle='-')
    plt.fill_between(r_used, G_pool-np.sqrt(VG_pool), G_pool+np.sqrt(VG_pool), color='grey', alpha=0.4)
    plt.legend()
    plt.title(r"Pooled Cross $G_{i,j}(r)$ Function")

    plt.subplot(2, 2, 4)
    r_used, K_pool, VK_pool, lamb_pool = pool([test_ppp, test_pp2, test_pp3], fun=Kcross, i='lymphocyte', j='tumor')
    # theoretic formula for K function is \pi*r^2
    K_theo = np.pi*np.array(r_used)**2
    plt.plot(r_used, K_pool, c='black', linewidth=3, linestyle='-', label=r"$K_{pooled}(r)$")
    plt.plot(r_used, K_theo, c='red', linewidth=3, linestyle='--', label=r"$K_{Poisson}(r)$")
    plt.plot(r_used, K_pool-np.sqrt(VK_pool), c='grey', linewidth=1, linestyle='-', label="K(r) Confidence Band")
    plt.plot(r_used, K_pool+np.sqrt(VK_pool), c='grey', linewidth=1, linestyle='-')
    plt.fill_between(r_used, K_pool-np.sqrt(VK_pool), K_pool+np.sqrt(VK_pool), color='grey', alpha=0.4)
    plt.legend()
    plt.title(r"Pooled Kcross $K_{i,j}(r)$ Function")

