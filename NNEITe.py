#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                  Neural Network assisted Euler Integrator                   #
#            Contact yhuang@caltech.edu for details. 06/08/2021               #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Data points are saved as adaptive-time-step given the stiffness of the ODEs #
#         NOTE: Only autonomous system is considered in this solver.          #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import numpy as np

class NNEIT():
    def __init__(self,options):
        self.nstp = 0
        self.nacc = 0
        self.nrej = 0
        
        self.rejlast = False
        self.rejmore = False
        self.hmin = options['Hmin']
        self.hmax = options['Hmax']
        self.hstart = options['Hstart']
        self.θmin = 1.00017e-04 # 1.14071524448533531626660095e-8
        self.eps = options['eps']
        self.maxstep = options['MaxStep']
        
    ''' This function works just for automonous system '''
    def __call__(self,ODEfcn,Tspan,Y0,ODEargs,NNweights):
        t = []
        y = []
        e = []
        
        Tstart,Tend = Tspan
        # two-element list, since it is adaptive time step
        
        T = Tstart
        H = self.hstart # initial step
        Y = Y0*1 # copy an array, Y0.copy() also works
        
        t.append(T)
        y.append(Y)
        while Tend-T >= self.eps:
            if self.nstp > self.maxstep:
                break
            if T+0.1*H==T:
                print('Time step too small, has to quit')
                break
            
            H = min(H,Tend-T)
            
            T,Y,H,E = self.findh(H,T,Y,ODEfcn,ODEargs,NNweights)
            t.append(T)
            y.append(Y)
            e.append(E)
            # print(T/abs(Tend-Tstart))
        self.t = np.array(t)
        self.y = np.array(y).T
        self.e = np.array(e).T
        return self
    
    '''Find the proper time step'''
    def findh(self,h,T,Y,ODEfcn,ODEargs,NNweights):
        while True: # Until h accepted
            if NNweights['func']==1: # will be updated in the future
                neuralnetwork = self.NNH2O2
            elif NNweights['func']==2:
                neuralnetwork = self.NNVerwer
            Yν,nerr,err = neuralnetwork(h,Y,ODEfcn,ODEargs,NNweights)
            self.nstp += 1
            fac = min(10,max(0.1,nerr**(-1/3)))
            hnew = h*fac
            if (nerr<=1) or (h<=self.hmin): # accept step
                self.nacc += 1
                Y = Yν
                T += h
                hnew = max(self.hmin,min(hnew,self.hmax))
                if self.rejlast: # no step size increase after rejection
                    hnew = min(hnew,h)
                self.rejlast, self.rejmore = False, False
                h = hnew
                break
            else: # reject step
                if self.rejmore:
                    hnew = h*0.1
                self.rejmore = self.rejlast
                self.rejlast = True
                h = hnew
                self.nrej += 1
        return T,Y,h,err
    
    '''Calculate νc, nerr, and err for the current step'''
    def NNH2O2(self,h,Y,ODEfcn,ODEargs,NNweights):
        M_1e6 = ODEargs['M']/ODEargs['cvt'][0]
        ci = Y*1
        ci[Y<=0] = 1.0e-100 # set non-positive conc to very small value
        ss,ps,ls = ODEfcn(ci,**ODEargs) # ppm min-1
        ls[ls<1.0e-80] = 1.0e-120
        ps[ps<1.0e-80] = 1.0e-120
        
        logcs = np.log(ci)+np.log(M_1e6) # ppm -> molec cm-3
        logls = np.log(ls) # ppm
        logps = np.log(ps) # ppm
        logdt = np.log(h) # min
        
        θ = ls*h/ci
        θ[θ<self.θmin] = self.θmin # some issue with values < θmin
        ω = (1-np.exp(-θ))/θ
        
        compl = np.hstack([logps,logls])
        logpl = np.tanh(logps-logls)
        comb  = np.hstack([logpl,ω])
        
        wc1, wc2 = NNweights['wc11'],NNweights['wc12']
        wt1, wt2 = NNweights['wt11'],NNweights['wt12']
        f = NNweights['scale1']
        
        ωn,ft = self.dydtfactor(wc1,wc2,ω,logls,logcs,logdt,compl)
        νc = ci + h*ss*ωn*(1+ft) # ppm
        νc[νc<=0] = 1.0e-100
        
        comb = np.tanh(wt1.T.dot(comb))
        comb = np.tanh(wt2.T.dot(comb))
        err  = comb*f # Ce/Ctol
        nerr = (np.sum(err**2)/len(Y))**0.5
        return νc,nerr,err
    
    '''Calculate νc, nerr, and err for the current step'''
    def NNVerwer(self,h,Y,ODEfcn,ODEargs,NNweights):
        M_1e6 = ODEargs['M']/ODEargs['cvt'][0]
        ci = Y*1
        ci[Y<=0] = 1.0e-100 # set non-positive conc to very small value
        ss,ps,ls = ODEfcn(ci,**ODEargs) # ppm min-1
        ls[ls<1.0e-80] = 1.0e-120
        ps[ps<1.0e-80] = 1.0e-120
        
        logcs = np.log(ci)+np.log(M_1e6) # ppm -> molec cm-3
        logls = np.log(ls) # ppm
        logps = np.log(ps) # ppm
        logdt = np.log(h) # min
        
        θ = ls*h/ci
        θ[θ<self.θmin] = self.θmin # some issue with values < θmin
        ω = (1-np.exp(-θ))/θ
        
        logpl = np.tanh(logps-logls)
        comb  = np.hstack([logpl,ω])
        if h < NNweights['r1']:
            wc1, wc2 = NNweights['wc11'], NNweights['wc12']
            wt1, wt2 = NNweights['wt11'], NNweights['wt12']
            f = NNweights['scale1']
        elif NNweights['r1'] <= h < NNweights['r2']:
            wc1, wc2 = NNweights['wc21'], NNweights['wc22']
            wt1, wt2 = NNweights['wt21'], NNweights['wt22']
            f = NNweights['scale2']
        elif NNweights['r2'] <= h < NNweights['r3']:
            wc1, wc2 = NNweights['wc31'], NNweights['wc32']
            wt1, wt2 = NNweights['wt31'], NNweights['wt32']
            f = NNweights['scale3']
        elif NNweights['r3'] <= h < NNweights['r4']:
            wc1, wc2 = NNweights['wc41'], NNweights['wc42']
            wt1, wt2 = NNweights['wt41'], NNweights['wt42']
            f = NNweights['scale4']
        else:
            wc1, wc2 = NNweights['wc51'], NNweights['wc52']
            wt1, wt2 = NNweights['wt51'], NNweights['wt52']
            f = NNweights['scale5']
        
        ωn,ft = self.dydtfactor(wc1,wc2,ω,logls,logcs,logdt,logpl)
        νc = ci + h*ss*ωn*(1+ft) # ppm
        νc[νc<=0] = 1.0e-100
        
        comb = np.tanh(wt1.T.dot(comb))
        comb = np.tanh(wt2.T.dot(comb))
        err  = comb*f # Ce/Ctol
        nerr = (np.sum(err**2)/len(Y))**0.5
        return νc,nerr,err
    
    def myelu(self,x,α=1.0):
        y = x*1.0
        y[x<0] = α*(np.exp(x[x<0])-1)
        return y
    
    def dydtfactor(self,wc1,wc2,ω,logls,logcs,logdt,logpl):
        if len(wc1)==0:
            ωn = ω
        else:
            logθn = np.hstack([logls,logcs,logdt])
            θn = np.exp(wc1.T.dot(logθn))
            θn[θn<self.θmin] = self.θmin # some issue with values < θmin
            ωn = (1-np.exp(-θn))/θn
        ft = self.myelu(wc2.T.dot(logpl))
        return ωn,ft

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               END OF NNEIT                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
