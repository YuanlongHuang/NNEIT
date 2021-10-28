#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# This python script is transformed from kpp-2.2.3/int/rosenbrock.f by yhuang #
#            Contact yhuang@caltech.edu for details. 12/08/2020               #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#      Solves the system y'=F(t,y) using a Rosenbrock method defined by:      #
#                                                                             #
#       G = 1/(H*gamma(1)) - ode_Jac(t0,Y0)                                   #
#       T_i = t0 + Alpha(i)*H                                                 #
#       Y_i = Y0 + \sum_{j=1}^{i-1} A(i,j)*K_j                                #
#       G * K_i = ode_Fun( T_i, Y_i ) + \sum_{j=1}^{i-1} C(i,j)/H * K_j +     #
#                    H*gamma(i)*dF/dT(t0, Y0)                                 #
#       Y1 = Y0 + \sum_{j=1}^S M(j)*K_j                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Data points are saved as adaptive-time-step given the stiffness of the ODEs #
#         NOTE: Only autonomous system is considered in this solver.          #
#               JACOBIAN MATRIX IS REQUIRED!!                                 #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import numpy as np
import sys
import scipy.linalg as la
from scipy.sparse import linalg
import scipy.sparse as sp

options = {
    'solver' : 2,
    'Hmin'   : 0.0e+00,
    'Hmax'   : 1.0e+01,
    'Hstart' : 1.0e-3,
    'Dmin'   : 1.0e-6,
    'FacMin' : 0.2,
    'FacMax' : 6.0,
    'FacSafe': 0.9,
    'FacRej' : 0.1,
    'AbsTol' : 1.0e-03,
    'RelTol' : 1.0e-04,
    'MaxStep': 2.0e+05,
    'eps'    : sys.float_info.epsilon,
}
# factor = dict(filter(lambda i: 'Fac' in i[0],options.items()))

class ROSsolver():
    def __init__(self,opt=options):
        params = self.ROSinitparams(opt['solver'])
        '''solver -- 1: Ros2, 2: Ros3, 3: Ros4, 4: Rodas3, 5: Rodas4'''
        keystr = ['rosMethod', 'ros_Name', 'ros_S', 'ros_A', 'ros_C', 'ros_M',\
                  'ros_E', 'ros_Alpha', 'ros_Gamma', 'ros_ELO', 'ros_NewF']
        self.p = dict(zip(keystr,params))
        
        '''Step size can be changed FacMin < Hnew/Hexit < FacMax'''
        Fmin   = abs(opt['FacMin']) if opt['FacMin'] else 0.2e0
        Fmax   = abs(opt['FacMax']) if opt['FacMax'] else 6.0e0
        '''FacSafe: Safety Factor in the computation of new step size'''
        Fsafe  = abs(opt['FacSafe']) if opt['FacSafe'] else 0.9e0
        '''FacRej: Factor to decrease step after 2 succesive rejections'''
        Frej   = abs(opt['FacRej']) if opt['FacRej'] else 0.1e0
        self.f = dict(zip(['min','max','saf','rej'],[Fmin,Fmax,Fsafe,Frej]))
        
        self.ISTAT = {'Nfun':0,'Njac':0,'Nstp':0,'Nacc':0,'Nrej':0,'Ninv':0}
        '''No. of function calls, jacobian calls, steps, accepted steps
                  rejected steps (except at very beginning), matrix inv'''
        
        self.RejectLastH = False
        self.RejectMoreH = False
        self.options = opt
        
    ''' This function works just for automonous system '''
    def __call__(self,ODEfcn,Tspan,Y0,ODEjac,ODEargs):
        t = []
        y = []
        k = []
        e = []
        
        Tstart,Tend = Tspan
        # two-element list, since it is adaptive time step
        
        opt = self.options
        # Lower bound on the step size: (non-negative value)
        Hmin = max(0.0,abs(opt['Hmin']))
        # Upper bound on the step size: (positive value)
        Hmax = min(abs(opt['Hmax']),abs(Tend-Tstart)) \
        if opt['Hmax'] else abs(Tend-Tstart)
        # Starting step size: (positive value)
        Hstart = min(abs(opt['Hstart']),abs(Tend-Tstart)) \
        if opt['Hstart'] else max(Hmin,opt['Dmin'])
        
        Direction = 1 if Tend >= Tstart else -1 # H = Direction*H
        T = Tstart
        H = min(max(abs(Hmin), abs(Hstart)), abs(Hmax))
        if abs(H) <= 10.0*opt['eps']:
            H = opt['Dmin']
        self.Hset = {'Hmin':Hmin,'Hmax':Hmax,'Hstart':H}
        
        Y = Y0*1 # copy an array, Y0.copy() also works
        # Y = Y0[:] # copy a list, list(Y0) also works
        t.append(T)
        y.append(Y)
        while (Tend-T)*Direction >= opt['eps']:
            if self.ISTAT['Nstp'] > opt['MaxStep']:
                break
            if T+0.1*H==T:
                print('Time step too small, has to quit')
                break
            
            H = min(H,abs(Tend-T))
            
            fcn0 = ODEfcn(T,Y,**ODEargs) # ODEfcn should return array
            self.ISTAT['Nfun'] += 1
            
            jac0 = ODEjac(T,Y,**ODEargs) # ODEjac should return array
            self.ISTAT['Njac'] += 1
            
            T,Y,H,K,E = self.ROSfindH(H,Direction,T,Y,fcn0,jac0,ODEfcn,ODEargs)
            t.append(T)
            y.append(Y)
            k.append(K)
            e.append(E)
            # print(T/abs(Tend-Tstart))
        self.t = np.array(t)
        self.y = np.array(y).T
        self.k = np.array(k).T
        self.e = np.array(e).T
        return self
    
    '''Find the proper time step'''
    def ROSfindH(self,H,Direction,T,Y,fcn0,jac0,ODEfcn,ODEargs):
        # AbsTol = [self.options['AbsTol']]*len(Y)
        # RelTol = [self.options['RelTol']]*len(Y)
        # keep the idea of various AbsTol for different compounds
        
        while True: # Until H accepted
            K = self.ROScalcK(H*Direction,T,Y,fcn0,jac0,ODEfcn,ODEargs)
            self.ISTAT['Nfun'] += sum(self.p['ros_NewF'])-1
            self.ISTAT['Ninv'] += self.p['ros_S']
            
            # Compute the new solution
            Ynew = Y*1
            for j in range(self.p['ros_S']):
                Ynew += self.p['ros_M'][j] * K[j]
            
            # Compute the error estimation
            Yerr = np.zeros(Y.shape)
            for j in range(self.p['ros_S']):
                Yerr += self.p['ros_E'][j] * K[j]
            Nerr,E = self.ROSnormErr(Y,Ynew,Yerr)
            
            # New step size is bounded by FacMin <= Hnew/H <= FacMax
            Fac  = min(self.f['max'],max(self.f['min'],self.f['saf']/\
                                         (Nerr**(1/self.p['ros_ELO']))))
            Hnew = H*Fac
            
            # Check the error magnitude and adjust step size
            self.ISTAT['Nstp'] += 1
            if  (Nerr <= 1) or (H <= self.Hset['Hmin']): # Accept step
                self.ISTAT['Nacc'] += 1
                Y = Ynew
                T += Direction*H
                Hnew = max(self.Hset['Hmin'], min(Hnew, self.Hset['Hmax']))
                if self.RejectLastH:
                    # No step size increase after a rejected step
                    Hnew = min(Hnew, H)
                self.RejectLastH = False
                self.RejectMoreH = False
                H = Hnew
                break # EXIT THE LOOP: WHILE STEP NOT ACCEPTED
            else: # Reject step
                if self.RejectMoreH:
                    Hnew = H*self.f['rej']
                self.RejectMoreH = self.RejectLastH
                self.RejectLastH = True
                H = Hnew
                self.ISTAT['Nrej'] += 1
        return T,Y,H,K,E
    
    '''Calculate K for the current step, return a list'''
    def ROScalcK(self,DirectionH,T,Y,fcn0,jac0,ODEfcn,ODEargs):
        K = list(range(self.p['ros_S'])) # K element should be array
        
        igh_j = sp.identity(len(Y))/(DirectionH*self.p['ros_Gamma'][0]) - jac0
        # I/(gamma*h)-jac
        
        # For the 1st istage the function has been computed previously
        fcn = fcn0*1
        RHS = fcn
        K[0] = sp.linalg.spsolve(igh_j, RHS)
        # The coefficient matrices A and C are strictly lower triangular.
        # The lower triangular (subdiagonal) elements are stored in row-wise
        # order: A(2,1) = ros_A[0], A(3,1)=ros_A[1], A(3,2)=ros_A[2], etc.
        # The general mapping formula is:
        #       A(i,j) = ros_A[ (i-1)*(i-2)/2 + j - 1 ]
        #       C(i,j) = ros_C[ (i-1)*(i-2)/2 + j - 1 ]
        for istage in range(2,self.p['ros_S']+1):
            # istage > 1 and a new function evaluation is needed
            if self.p['ros_NewF'][istage-1]:
                Ynew = Y*1
                for j in range(istage-1): # note x += 1 and x = x + 1 mutable
                    Ynew += self.p['ros_A'][(istage-1)*(istage-2)//2+j]*K[j]
                    # //2 indicates floor division in Python 3.x
                Tau = T + self.p['ros_Alpha'][istage-1]*DirectionH
                fcn = ODEfcn(Tau,Ynew,**ODEargs)
            RHS = fcn*1
            for j in range(istage-1):
                C_H = self.p['ros_C'][(istage-1)*(istage-2)//2+j]/DirectionH
                RHS = RHS + C_H*K[j]
            K[istage-1] = sp.linalg.spsolve(igh_j, RHS)
        return K
    
    '''Normailze error, return a scalor'''
    def ROSnormErr(self,Y,Ynew,Yerr):
        RelTol = [self.options['RelTol']]*len(Y)
        AbsTol_ = self.options['AbsTol']
        if type(AbsTol_)==float:
            AbsTol = [AbsTol_]*len(Y)
        Ymax = np.maximum(abs(Y),abs(Ynew)) # element-wise maximum
        Ytol = AbsTol + RelTol*Ymax # Y tolorance
        Ynor = Yerr/Ytol
        err  = (sum(Ynor**2)/len(Y))**0.5
        # idx  = Ytol > self.options['eps']
        # err  = (sum((Yerr[idx]/Ytol[idx])**2)/len(idx))**0.5
        Nerr = max(err,1.0e-10)
        return Nerr,Ynor
    
    '''Initialize solver parameters, return a list'''
    def ROSinitparams(self,x):
        return {
            1: self.Ros2(),
            2: self.Ros3(),
            3: self.Ros4(),
            4: self.Rodas3(),
            5: self.Rodas4(),
        }.get(x,self.Ros3()) # default solver -- Ros3
    
    
    '''One-Step Integration'''
    def ROS_TYK(self,H,T,Y,ODEfcn,ODEjac,ODEargs):
        fcn = ODEfcn(T,Y,**ODEargs) # ODEfcn should return array
        jac = ODEjac(T,Y,**ODEargs) # ODEjac should return array
        K = self.ROScalcK(H,T,Y,fcn,jac,ODEfcn,ODEargs)
        Y = Y*1
        for j in range(self.p['ros_S']):
            Y += self.p['ros_M'][j] * K[j]
        T += H
        return T,Y,K
    
    '''Five Rosenbrock methods, return a list'''
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # --- AN L-STABLE METHOD, 2 stages, order 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    def Ros2(self):
        rosMethod = 1
        # Name of the method
        ros_Name = 'ROS-2'
        # Number of stages
        ros_S = 2

        ros_A = list(range(ros_S*(ros_S-1)//2))
        ros_C = list(range(ros_S*(ros_S-1)//2))
        ros_M, ros_E = list(range(ros_S)), list(range(ros_S))
        ros_Alpha, ros_Gamma = list(range(ros_S)), list(range(ros_S))
        ros_NewF = list(range(ros_S))

        g = 1.0 + 1.0/2.0**0.5
        # A_i = Coefficients for Ynew
        ros_A[0] = (1.0)/g
        # C_i = Coefficients for RHS K_j
        ros_C[0]= (-2.0)/g
        # M_i = Coefficients for new step solution
        ros_M[0] = (3.0)/(2.0*g)
        ros_M[1] = (1.0)/(2.0*g)
        # E_i = Coefficients for error estimator       
        ros_E[0] = 1.0/(2.0*g)
        ros_E[1] = 1.0/(2.0*g)
        # Y_stage_i = Y( T + H*Alpha_i )
        ros_Alpha[0] = 0.0
        ros_Alpha[1] = 1.0
        # Gamma_i = \sum_j^i gamma_{i,j}, Coefficients of t-derivative
        ros_Gamma[0] =  g
        ros_Gamma[1] = -g
        # ros_ELO = estimator of local order, the minimum between the main and
        #           the embedded scheme orders + 1
        ros_ELO = 2.0
        # Does the stage i require a new function evaluation (ros_NewF[i]=TRUE)
        #  or re-use the function evaluation from stage i-1 (ros_NewF[i]=FALSE)
        ros_NewF[0] = True
        ros_NewF[1] = True

        return [rosMethod, ros_Name, ros_S, ros_A, ros_C, ros_M, ros_E,\
                ros_Alpha, ros_Gamma, ros_ELO, ros_NewF]
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # --- AN L-STABLE METHOD, 3 stages, order 3, 2 function evaluations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    def Ros3(self):
        rosMethod = 2
        # Name of the method
        ros_Name = 'ROS-3'
        # Number of stages
        ros_S = 3

        ros_A = list(range(ros_S*(ros_S-1)//2))
        ros_C = list(range(ros_S*(ros_S-1)//2))
        ros_M, ros_E = list(range(ros_S)), list(range(ros_S))
        ros_Alpha, ros_Gamma = list(range(ros_S)), list(range(ros_S))
        ros_NewF = list(range(ros_S))

        # A_i = Coefficients for Ynew
        ros_A[0] = 1.0
        ros_A[1] = 1.0
        ros_A[2] = 0.0
        # C_i = Coefficients for RHS K_j
        ros_C[0] = -0.10156171083877702091975600115545e+01
        ros_C[1] =  0.40759956452537699824805835358067e+01
        ros_C[2] =  0.92076794298330791242156818474003e+01
        # M_i = Coefficients for new step solution
        ros_M[0] =  0.1e+01
        ros_M[1] =  0.61697947043828245592553615689730e+01
        ros_M[2] = -0.42772256543218573326238373806514
        # E_i = Coefficients for error estimator       
        ros_E[0] =  0.5
        ros_E[1] = -0.29079558716805469821718236208017e+01
        ros_E[2] =  0.22354069897811569627360909276199
        # Y_stage_i = Y( T + H*Alpha_i )
        ros_Alpha[0] = 0.0
        ros_Alpha[1] = 0.43586652150845899941601945119356
        ros_Alpha[2] = 0.43586652150845899941601945119356
        # Gamma_i = \sum_j^i gamma_{i,j}, Coefficients of t-derivative
        ros_Gamma[0] = 0.43586652150845899941601945119356
        ros_Gamma[1] = 0.24291996454816804366592249683314
        ros_Gamma[2] = 0.21851380027664058511513169485832e+01
        # ros_ELO = estimator of local order, the minimum between the main and
        #           the embedded scheme orders + 1
        ros_ELO = 3.0
        # Does the stage i require a new function evaluation (ros_NewF[i]=TRUE)
        #  or re-use the function evaluation from stage i-1 (ros_NewF[i]=FALSE)
        ros_NewF[0] = True
        ros_NewF[1] = True
        ros_NewF[2] = False

        return [rosMethod, ros_Name, ros_S, ros_A, ros_C, ros_M, ros_E,\
                ros_Alpha, ros_Gamma, ros_ELO, ros_NewF]
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #     L-STABLE ROSENBROCK METHOD OF ORDER 4, WITH 4 STAGES
    #     L-STABLE EMBEDDED ROSENBROCK METHOD OF ORDER 3
    #
    #      E. HAIRER AND G. WANNER, SOLVING ORDINARY DIFFERENTIAL
    #      EQUATIONS II. STIFF AND DIFFERENTIAL-ALGEBRAIC PROBLEMS.
    #      SPRINGER SERIES IN COMPUTATIONAL MATHEMATICS,
    #      SPRINGER-VERLAG (1990)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    def Ros4(self):
        rosMethod = 3
        # Name of the method
        ros_Name = 'ROS-4'
        # Number of stages
        ros_S = 4

        ros_A = list(range(ros_S*(ros_S-1)//2))
        ros_C = list(range(ros_S*(ros_S-1)//2))
        ros_M, ros_E = list(range(ros_S)), list(range(ros_S))
        ros_Alpha, ros_Gamma = list(range(ros_S)), list(range(ros_S))
        ros_NewF = list(range(ros_S))

        # A_i = Coefficients for Ynew
        ros_A[0] = 0.2000000000000000e+01
        ros_A[1] = 0.1867943637803922e+01
        ros_A[2] = 0.2344449711399156
        ros_A[3] = ros_A[1]
        ros_A[4] = ros_A[2]
        ros_A[5] = 0.0
        # C_i = Coefficients for RHS K_j
        ros_C[0] = -0.7137615036412310e+01
        ros_C[1] =  0.2580708087951457e+01
        ros_C[2] =  0.6515950076447975
        ros_C[3] = -0.2137148994382534e+01
        ros_C[4] = -0.3214669691237626
        ros_C[5] = -0.6949742501781779
        # M_i = Coefficients for new step solution
        ros_M[0] = 0.2255570073418735e+01
        ros_M[1] = 0.2870493262186792
        ros_M[2] = 0.4353179431840180
        ros_M[3] = 0.1093502252409163e+01
        # E_i = Coefficients for error estimator       
        ros_E[0] = -0.2815431932141155
        ros_E[1] = -0.7276199124938920e-01
        ros_E[2] = -0.1082196201495311
        ros_E[3] = -0.1093502252409163e+01
        # Y_stage_i = Y( T + H*Alpha_i )
        ros_Alpha[0] = 0.0
        ros_Alpha[1] = 0.1145640000000000e+01
        ros_Alpha[2] = 0.6552168638155900
        ros_Alpha[3] = ros_Alpha[2]
        # Gamma_i = \sum_j^i gamma_{i,j}, Coefficients of t-derivative
        ros_Gamma[0] =  0.5728200000000000
        ros_Gamma[1] = -0.1769193891319233e+01
        ros_Gamma[2] =  0.7592633437920482
        ros_Gamma[3] = -0.1049021087100450
        # ros_ELO = estimator of local order, the minimum between the main and
        #           the embedded scheme orders + 1
        ros_ELO  = 4.0
        # Does the stage i require a new function evaluation (ros_NewF[i]=TRUE)
        #  or re-use the function evaluation from stage i-1 (ros_NewF[i]=FALSE)
        ros_NewF[0]  = True
        ros_NewF[1]  = True
        ros_NewF[2]  = True
        ros_NewF[3]  = False

        return [rosMethod, ros_Name, ros_S, ros_A, ros_C, ros_M, ros_E,\
                ros_Alpha, ros_Gamma, ros_ELO, ros_NewF]
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # --- A STIFFLY-STABLE METHOD, 4 stages, order 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    def Rodas3(self):
        rosMethod = 4
        # Name of the method
        ros_Name = 'RODAS-3'
        # Number of stages
        ros_S = 4

        ros_A = list(range(ros_S*(ros_S-1)//2))
        ros_C = list(range(ros_S*(ros_S-1)//2))
        ros_M, ros_E = list(range(ros_S)), list(range(ros_S))
        ros_Alpha, ros_Gamma = list(range(ros_S)), list(range(ros_S))
        ros_NewF = list(range(ros_S))

        # A_i = Coefficients for Ynew
        ros_A[0] = 0.0
        ros_A[1] = 2.0
        ros_A[2] = 0.0
        ros_A[3] = 2.0
        ros_A[4] = 0.0
        ros_A[5] = 1.0
        # C_i = Coefficients for RHS K_j
        ros_C[0] =  4.0
        ros_C[1] =  1.0
        ros_C[2] = -1.0
        ros_C[3] =  1.0
        ros_C[4] = -1.0
        ros_C[5] = -(8.0/3.0)
        # M_i = Coefficients for new step solution
        ros_M[0] = 2.0
        ros_M[1] = 0.0
        ros_M[2] = 1.0
        ros_M[3] = 1.0
        # E_i = Coefficients for error estimator       
        ros_E[0] = 0.0
        ros_E[1] = 0.0
        ros_E[2] = 0.0
        ros_E[3] = 1.0
        # Y_stage_i = Y( T + H*Alpha_i )
        ros_Alpha[0] = 0.0
        ros_Alpha[1] = 0.0
        ros_Alpha[2] = 1.0
        ros_Alpha[3] = 1.0
        # Gamma_i = \sum_j^i gamma_{i,j}, Coefficients of t-derivative
        ros_Gamma[0] = 0.5
        ros_Gamma[1] = 1.5
        ros_Gamma[2] = 0.0
        ros_Gamma[3] = 0.0
        # ros_ELO = estimator of local order, the minimum between the main and
        #           the embedded scheme orders + 1
        ros_ELO  = 3.0
        # Does the stage i require a new function evaluation (ros_NewF[i]=TRUE)
        #  or re-use the function evaluation from stage i-1 (ros_NewF[i]=FALSE)
        ros_NewF[0]  = True
        ros_NewF[1]  = False
        ros_NewF[2]  = True
        ros_NewF[3]  = True

        return [rosMethod, ros_Name, ros_S, ros_A, ros_C, ros_M, ros_E,\
                ros_Alpha, ros_Gamma, ros_ELO, ros_NewF]
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #     STIFFLY-STABLE ROSENBROCK METHOD OF ORDER 4, WITH 6 STAGES
    #
    #      E. HAIRER AND G. WANNER, SOLVING ORDINARY DIFFERENTIAL
    #      EQUATIONS II. STIFF AND DIFFERENTIAL-ALGEBRAIC PROBLEMS.
    #      SPRINGER SERIES IN COMPUTATIONAL MATHEMATICS,
    #      SPRINGER-VERLAG (1996)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    def Rodas4(self):
        rosMethod = 5
        # Name of the method
        ros_Name = 'RODAS-4'
        # Number of stages
        ros_S = 6

        ros_A = list(range(ros_S*(ros_S-1)//2))
        ros_C = list(range(ros_S*(ros_S-1)//2))
        ros_M, ros_E = list(range(ros_S)), list(range(ros_S))
        ros_Alpha, ros_Gamma = list(range(ros_S)), list(range(ros_S))
        ros_NewF = list(range(ros_S))

        # A_i = Coefficients for Ynew
        ros_A[0]  =  0.1544000000000000e+01
        ros_A[1]  =  0.9466785280815826
        ros_A[2]  =  0.2557011698983284
        ros_A[3]  =  0.3314825187068521e+01
        ros_A[4]  =  0.2896124015972201e+01
        ros_A[5]  =  0.9986419139977817
        ros_A[6]  =  0.1221224509226641e+01
        ros_A[7]  =  0.6019134481288629e+01
        ros_A[8]  =  0.1253708332932087e+02
        ros_A[9]  = -0.6878860361058950
        ros_A[10] =  ros_A[6]
        ros_A[11] =  ros_A[7]
        ros_A[12] =  ros_A[8]
        ros_A[13] =  ros_A[9]
        ros_A[14] =  1.0
        # C_i = Coefficients for RHS K_j
        ros_C[0]  = -0.5668800000000000e+01
        ros_C[1]  = -0.2430093356833875e+01
        ros_C[2]  = -0.2063599157091915
        ros_C[3]  = -0.1073529058151375
        ros_C[4]  = -0.9594562251023355e+01
        ros_C[5]  = -0.2047028614809616e+02
        ros_C[6]  =  0.7496443313967647e+01
        ros_C[7]  = -0.1024680431464352e+02
        ros_C[8]  = -0.3399990352819905e+02
        ros_C[9]  =  0.1170890893206160e+02
        ros_C[10] =  0.8083246795921522e+01
        ros_C[11] = -0.7981132988064893e+01
        ros_C[12] = -0.3152159432874371e+02
        ros_C[13] =  0.1631930543123136e+02
        ros_C[14] = -0.6058818238834054e+01
        # M_i = Coefficients for new step solution
        ros_M[0] = ros_A[6]
        ros_M[1] = ros_A[7]
        ros_M[2] = ros_A[8]
        ros_M[3] = ros_A[9]
        ros_M[4] = 1.0
        ros_M[5] = 1.0
        # E_i = Coefficients for error estimator       
        ros_E[0] = 0.0
        ros_E[1] = 0.0
        ros_E[2] = 0.0
        ros_E[3] = 0.0
        ros_E[4] = 0.0
        ros_E[5] = 1.0
        # Y_stage_i = Y( T + H*Alpha_i )
        ros_Alpha[0] = 0.000
        ros_Alpha[1] = 0.386
        ros_Alpha[2] = 0.210
        ros_Alpha[3] = 0.630
        ros_Alpha[4] = 1.000
        ros_Alpha[5] = 1.000
        # Gamma_i = \sum_j^i gamma_{i,j}, Coefficients of t-derivative
        ros_Gamma[0] =  0.2500000000000000
        ros_Gamma[1] = -0.1043000000000000
        ros_Gamma[2] =  0.1035000000000000
        ros_Gamma[3] = -0.3620000000000023e-01
        ros_Gamma[4] =  0.0
        ros_Gamma[5] =  0.0
        # ros_ELO = estimator of local order, the minimum between the main and
        #           the embedded scheme orders + 1
        ros_ELO = 4.0
        # Does the stage i require a new function evaluation (ros_NewF[i]=TRUE)
        #  or re-use the function evaluation from stage i-1 (ros_NewF[i]=FALSE)
        ros_NewF[0] = True
        ros_NewF[1] = True
        ros_NewF[2] = True
        ros_NewF[3] = True
        ros_NewF[4] = True
        ros_NewF[5] = True

        return [rosMethod, ros_Name, ros_S, ros_A, ros_C, ros_M, ros_E,\
                ros_Alpha, ros_Gamma, ros_ELO, ros_NewF]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                    END OF ROSENBROCK STIFF ODE SOLVER                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
