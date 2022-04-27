import numpy as np

def div(cx,cy):
    """
    cy and cy are coordonates of a vector field.
    #the function computes the discrete divergence of this vector field
    """

    nr,nc=cx.shape

    ddx=np.zeros((nr,nc))
    ddy=np.zeros((nr,nc))

    ddx[:,1:-1]=cx[:,1:-1]-cx[:,0:-2]
    ddx[:,0]=cx[:,0]
    ddx[:,-1]=-cx[:,-2]
  
    ddy[1:-1,:]=cy[1:-1,:]-cy[0:-2,:]
    ddy[0,:]=cy[0,:]
    ddy[-1,:]=-cy[-2,:]
 
    d=ddx+ddy

    return d


def grad(im):
    """
    computes the gradient of the image 'im'
    """

    # image size 
    nr,nc=im.shape
  
    gx = im[:,1:]-im[:,0:-1]
    gx = np.block([gx,np.zeros((nr,1))])

    gy =im[1:,:]-im[0:-1,:]
    gy=np.block([[gy],[np.zeros((1,nc))]])
    return gx,gy


def laplacian(im):
    """
    computes the laplacian of the image 'im'
    """

    cx, cy = grad(im)
    return div(cx, cy)


def inpainting(v, epsilon, p, niter, mask):
    """
    Parameters:
    v: degraded image
    epislon: small value to avoid division by 0
    p: gradient step 
    niter: number of iterations 
    mask: mask of missing pixels.
    """
    
    u = np.copy(v)

    for i in range(niter):
        ux,uy  = grad(u)
        normgrad = np.sqrt(ux**2 + uy**2 + epsilon**2)
        u = u - p * ( - div(ux/normgrad,uy/normgrad))
        u[mask] = v[mask]
    
    return u


def convol_aperiodic(a,b):
    return scipy.signal.convolve2d(a,b, boundary='symm', mode='same')


def tvdeconv(ub,k,lambd,niter):
    """
    Deconvolution with double splitting and known kernel
    """

    # kernel
    k = k/np.sum(k)
    normk = np.sum(np.abs(k))
    kstar = k[::-1,::-1]

    #initialization
    nr,nc = ub.shape
    ut = np.copy(ub)
    ubar = np.copy(ub)

    p = np.zeros((nr,nc,2))
    q = np.zeros((nr,nc))
    tau   = 0.9/np.sqrt(8*lambd**2 + normk**2)
    sigma = tau
    theta = 1
    
    #Double splitting
    for i in range(niter):
        #INSERT YOUR CODE HERE  

        # ProxF for p
        ux,uy  = grad(ubar)
        p = p + sigma*lambd*np.stack((ux,uy),axis=2)
        #normep = np.abs(p[:,:,0])+np.abs(p[:,:,1])
        normep = np.sqrt(p[:,:,0]**2+p[:,:,1]**2)
        normep = normep*(normep>1) + (normep<=1)
        p[:,:,0] = p[:,:,0]/normep
        p[:,:,1] = p[:,:,1]/normep

        # ProxF for q
        q = q + sigma*convol_aperiodic(ubar, k)
        q = (q-sigma*ub)/(1+sigma)

        # subgradient step on u
        d=div(p[:,:,0],p[:,:,1])
        unew = (ut + tau*lambd*d - tau*convol_aperiodic(q, kstar)) 
    
        #extragradient step on u 
        ubar = unew + theta*(unew-ut)
        ut = np.copy(unew) 

        #END INSERT YOUR CODE HERE  

    return ut


def chambolle_pock_prox_TV(TV,ub,lambd,niter, **opts):
    """
    the function solves the problem
    - TVL2
       argmin_u   1/2 \| u - ub\|^2 + \lambda TV(u)
    - or TVL2A
       argmin_u   1/2 \| Au - ub\|^2 + \lambda TV(u)
       with A = diagonal matrix represented by the mask send as an opt on parameters
    - or TVL1
       argmin_u   1/2 \| u - ub\|_1 + \lambda TV(u)
    with TV(u) = \sum_i \|\nabla u (i) \|_2
    uses niter iterations of Chambolle-Pock
    """

    nr,nc = ub.shape
    ut = np.copy(ub)
    ubar = np.copy(ut)
    p = np.zeros((nr,nc,2))
    tau = 0.9/np.sqrt(8*lambd**2)
    sigma = 0.9/np.sqrt(8*lambd**2) 
    theta = 1
    
    # for TVL2A case
    mask = opts.get('mask', np.ones_like(ub))
    
    for k in range(niter):
        # calcul de proxF
        ux,uy  = grad(ubar)
        p = p + sigma*lambd*np.stack((ux,uy),axis=2)
        normep = np.sqrt(p[:,:,0]**2+p[:,:,1]**2)
        normep = normep*(normep>1) + (normep<=1)
        p[:,:,0] = p[:,:,0]/normep
        p[:,:,1] = p[:,:,1]/normep

        # calcul de proxG
        d=div(p[:,:,0],p[:,:,1])
        if (TV == "TVL2"):
            unew = 1/(1+tau)*(ut+tau*lambd*d+tau*ub) 
        elif (TV == "TVL2A"):    
            unew = 1/(1+tau*mask)*(ut+tau*lambd*d+tau*mask*ub)
        else:
            uaux = ut+tau*lambd*d
            unew = (uaux-tau)*(uaux-ub>tau)+(uaux+tau)*(uaux-ub<-tau)+ub*(abs(uaux-ub)<=tau)
        
        #extragradient step
        ubar = unew+theta*(unew-ut)
        ut = np.copy(unew)
           
    return ut