# use sparse matrix to calculate rates of change and Jacobian matrix
# contact yhuang@caltech.edu for details, 12/2020

import numpy as np
import scipy.sparse as sp

# changing rates, can be used independently
def mcm_dCdt(t,C,ks,rmat,pmat,smat,rRO2,iRO2,M,cvt):
    # C is a vector in ppm, C.shape = (s,)
    # ks is a vector, ks.shape = (n,)
    # rmat stores reaction info, (n,3)
    # pmat & smat are sparse matrices, (n,s) & (s,n)
    # rRO2 is a boolean vector of reaction, rRO2.shape = (n,)
    # iRO2 is a boolean vector of species index, iRO2.shape = (n,)
    # M is a scalar of molecule conc, molec cm-3
    # cvt is a vector for unit conversion
    cC = C/cvt[0]*M # 1 -> molec cm-3, (s,)
    rxn = ks*1 # rxn.shape = (n,)
    cRO2 = sum(cC[iRO2])
    rxn[rRO2] = cRO2*rxn[rRO2]
    rxn = rxn*cC[rmat[:,0]] # 1st order rxn
    idx = rmat[:,2]==2 # hete-molec 2nd order rxn
    rxn[idx] = rxn[idx]*cC[rmat[idx,1]]
    idx = rmat[:,2]==3 # homo-molec 2nd order rxn
    rxn[idx] = rxn[idx]*cC[rmat[idx,0]]
    dcdt = smat.dot(rxn)/M*cvt[0]*cvt[1]
    return dcdt # ppm min-1, dCdt.shape = (s,), molec cm-3 s-1

# production and loss rates, can be used independently
def mcm_pdls(C,ks,rmat,pmat,smat,rRO2,iRO2,M,cvt):
    # C is a vector in ppm, C.shape = (s,)
    # ks is a vector, ks.shape = (n,)
    # pmat & smat are sparse matrices, (n,s) & (s,n)
    # rRO2 is a boolean vector of reaction, rRO2.shape = (n,)
    # iRO2 is a boolean vector of species index, iRO2.shape = (n,)
    # M is a scalar of molecule conc, molec cm-3
    # cvt is a vector for unit conversion
    cC = C/cvt[0]*M # 1 -> molec cm-3, (s,)
    rxn = ks*1 # rxn.shape = (n,)
    cRO2 = sum(cC[iRO2])
    rxn[rRO2] = cRO2*rxn[rRO2]
    rxn = rxn*cC[rmat[:,0]] # 1st order rxn
    idx = rmat[:,2]==2 # hete-molec 2nd order rxn
    rxn[idx] = rxn[idx]*cC[rmat[idx,1]]
    idx = rmat[:,2]==3 # homo-molec 2nd order rxn
    rxn[idx] = rxn[idx]*cC[rmat[idx,0]]
    dcdt = smat.dot(rxn)/M*cvt[0]*cvt[1]
    prod = pmat.dot(rxn)/M*cvt[0]*cvt[1]
    loss = prod - dcdt
    return dcdt, prod, loss # ppm min-1, shape = (s,), molec cm-3 s-1

def mcm_Jacob(t,C,ks,rmat,pmat,smat,rRO2,iRO2,M,cvt):
    cC = C/cvt[0]*M # 1 -> molec cm-3, (s,)
    rxn = ks*1 # rxn.shape = (n,)
    cRO2 = sum(cC[iRO2])
    rxn[rRO2] = rxn[rRO2]*cRO2 + rxn[rRO2]*cC[rmat[rRO2,0]]
    idx = np.where(rmat[:,2]==1) # 1st order
    j1 = [rxn[idx].reshape(1,-1),idx,rmat[idx,0]]
    idx = np.where(rmat[:,2]==2) # hete-molec 2nd order
    j21 = [rxn[idx]*cC[rmat[idx,1]],idx,rmat[idx,0]]
    j22 = [rxn[idx]*cC[rmat[idx,0]],idx,rmat[idx,1]]
    idx = np.where(rmat[:,2]==3) # homo-molec 2nd order
    j23 = [rxn[idx]*2*cC[rmat[idx,0]],idx,rmat[idx,0]]
    j = list(zip(j1,j21,j22,j23))
    v = np.hstack(j[0]).flatten()
    r = np.hstack(j[1]).flatten()
    c = np.hstack(j[2]).flatten()
    dfdc = sp.csr_matrix((v,(r,c)),smat.T.shape)
    jac = smat.dot(dfdc)*cvt[1] # s-1 -> min-1
    return jac # dfdc.shape = (s,s)

def mcm_init(init_C,species,convt):
    # initialize C0
    s = len(species) # RO2 is not included in species
    C0 = np.zeros(s)
    for j in init_C.keys():
        C0[species.index(j)] = init_C[j]
    C0 = C0*convt[0]
    return C0

def ros3_err_sol(sol): # for ROSsolver w/o ynor as output
    # sol is the solution from ROS3 solver
    s = sol.y.shape[0]
    atol = sol.options['AbsTol']
    rtol = sol.options['RelTol']
    η = [ 0.5,
         -0.29079558716805469821718236208017e+01,
          0.22354069897811569627360909276199]
    k1 = sol.k[:,0,:]
    k2 = sol.k[:,1,:]
    k3 = sol.k[:,2,:]
    
    yerr = η[0]*k1+η[1]*k2+η[2]*k3
    ymax = np.maximum(sol.y[:,:-1],sol.y[:,1:])
    ytot = ymax*rtol+atol
    ynor = yerr/ytot
    esum = np.sum(ynor**2,0)
    frac = ynor**2/esum
    nerr = (esum/s)**0.5
    return nerr,ynor,frac

def ros3_err(C0,h,ODEfcn,ODEjac,ODEargs,rtol,atol):
    γ = 0.43586652150845899941601945119356
    α = [-0.10156171083877702091975600115545e+01,
          0.40759956452537699824805835358067e+01,
          0.92076794298330791242156818474003e+01]
    β = [ 0.1e+01,
          0.61697947043828245592553615689730e+01,
         -0.42772256543218573326238373806514]
    η = [ 0.5,
         -0.29079558716805469821718236208017e+01,
          0.22354069897811569627360909276199]
    s = len(C0)
    
    f0 = ODEfcn(0,C0,**ODEargs) # ppm min-1
    j = ODEjac(0,C0,**ODEargs).todense() # min-1
    jpre = np.eye(s)/h/γ-j
    k1 = np.linalg.solve(jpre,f0)
    f1 = ODEfcn(0,C0+k1,**ODEargs)
    k2 = np.linalg.solve(jpre,f1+α[0]*k1/h)
    k3 = np.linalg.solve(jpre,f1+α[1]*k1/h+α[2]*k2/h)
    
    deltay = β[0]*k1+β[1]*k2+β[2]*k3 # ppm
    νy = C0 + deltay
    my = np.maximum(abs(C0),abs(νy))
    ytol = atol + rtol*my
    yerr = η[0]*k1+η[1]*k2+η[2]*k3
    ynor = yerr/ytol
    esum = np.sum(ynor**2,0)
    frac = ynor**2/esum
    nerr = (esum/s)**0.5
    dydt = deltay/h # ppm min-1
    return k1,k2,k3,dydt,nerr,ynor,frac
