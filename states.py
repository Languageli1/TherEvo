import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.misc import derivative
import scipy

#rho_c = float(input("Input rho_c (kg/cm3):\n"))
rho_c = 6300 # density of Martian core, kg m-3
R_c = 1830 # CMB radius, km
P_cmb0 = 18   # CMB pressure, GPa
T_cmb0 = 1860 
k = 22
q_cmb0 = 0.9

Cp = 780     # 热容, J kg-1 K-1
e_s = 0.12   # S的质量分数
L = 7272     # km  
G = 6.6873*10**(-11) # m3 kg-1 s-2, 

#%%
P_param=np.array([ 5.40528557e+02,  0.00000000e+00, -7.26243043e+01,  5.42748223e-03,
        2.46141707e+00, -1.23887453e-04,  1.87440365e-07])
E_param=np.array([ 1.58330563e+00,  0.00000000e+00, -9.92000699e-01,  2.68648301e-04,
        3.54017091e-02,  1.34939551e-06,  5.05314332e-09])

#%% physical property
def solve_V(T,P,P_param):
    a = P_param[4]# x^2系数
    b = P_param[2]+ P_param[5]*T # x系数
    c = P_param[0]+ P_param[1]+ P_param[3]*T+ P_param[6]*T*T -P
    delta = b*b - 4*a*c
    if delta < 0:
        x1=0
    else:
        x1 = (-b-np.sqrt(delta))/2/a
    return x1

def KT(V,T,P_param):
    # isothermal bulk modulus, unit GPa
    return -V*(P_param[2] + 2*P_param[4]*V + P_param[5]*T)

def alpha(V,T,P_param):
    # thermal expansion, unit K-1
    return -(P_param[3]+P_param[5]*V+2*P_param[6]*T)/(P_param[2]+2*P_param[4]*V+P_param[5]*V)/V

def Cv(V,T,E_param):
    # heat capacity at constant volume, unit eV/K 
    return E_param[3]+E_param[5]*V+2*E_param[6]*T

def CPP(V,T,P_param,E_param):
    # unit eV/K
    fv = P_param[2]+P_param[4]*2*V+P_param[5]*T
    ft = P_param[5]*V+P_param[6]*2*T
    mv = E_param[2]+E_param[4]*2*V+E_param[5]*T
    mt = E_param[5]*V+E_param[6]*2*T
    p = P_param[0] + P_param[1] + P_param[2]*V + P_param[3]*T +P_param[4]*V*V + P_param[5]*V*T + P_param[6]*T*T 
    return  (mt+(mv-p)*(-ft/fv))*1.602*6.02*10**4 # J kg-1 K-1

def gamma(V,T,P_param,E_param):
    # Gruneisen number, no unit
    return alpha(V,T,P_param)*KT(V,T,P_param)*V/Cv(V,T,E_param)/160.2

def Ks(V,T,P_param,E_param):
    return KT(V,T,P_param)*(1+alpha(V,T,P_param)*gamma(V,T,P_param,E_param))

#%% Calculate g(r)
def g_diff(g,r):
    # g(r), m s-2；
    if r==0:
        dgdr = 4*np.pi*G*rho_c*10**(3)
    else:
        dgdr = 4*np.pi*G*rho_c*10**(3)-2*g/r
    return dgdr

def g_calc(r):
    gg = integrate.odeint(g_diff, (0), np.array([0,r]),) 
    return gg[1,0]

#%% Calculate T and P  
def TPg_diff(p,r,para):
    # differential equation of T/K, P/GPa and g/(m s-2)
    # r0=r_cmb-r
    P_param, E_param = para
    T,P,g = p.tolist() # 
    V = solve_V(T,P,P_param)
    if r==R_c:
        dgdr = -4*np.pi*G*rho_c*10**(3)
    else:
        dgdr = -4*np.pi*G*rho_c*10**(3)+2*g/(R_c-r) 
    dTdr = rho_c*g*gamma(V,T,P_param,E_param)*T*10**(-6)/Ks(V,T,P_param,E_param)
    dPdr = rho_c*g*10**(-6) # GPa/km 
    return dTdr, dPdr,dgdr

#%%
def TP_calc(r,Tcmb):
    # r is single number
    # return T(r), P(r) and g(r)
    ss = integrate.odeint(TPg_diff, (Tcmb,P_cmb0,g_calc(R_c)), [0,R_c-r],args=([P_param,E_param],)) # return ri, Tc    
    T = ss[1,0]
    P = ss[1,1]
    return P, T

# %% 两个离散数列的交点
def cross_roots(X1,Y1,Y2):
    Y_new = Y1 - Y2
    tck = interpolate.make_interp_spline(x=X1, y=Y_new, k=1)
    piecewise_polynomial = interpolate.PPoly.from_spline(tck, extrapolate=None)
    roots_X_ = piecewise_polynomial.roots()
    intersection_X = roots_X_[np.where(np.logical_and(roots_X_>=X1[0], roots_X_<=X1[-1]))]
    intersection_Y = np.interp(intersection_X, X1,Y1)
    return [intersection_X,intersection_Y]


def interaction(x,y1,y2):
    # 两个离散数列的交点，返回x值
    interp1 = interpolate.InterpolatedUnivariateSpline(x,y1)
    interp2 = interpolate.InterpolatedUnivariateSpline(x,y2)
    def difference(x):
        return np.abs(interp1(x)-interp2(x))
    x_at_crossing = scipy.optimize.fsolve(difference,x0=3.0)
    return x_at_crossing

#%%
# melting line 1 for 10.6 wt.% S
def Tm1(P):
    tm1 = 1990.5*(1-0.0022*P+3.8*10**(-7)*P*P)
    dtm1dp = 1990.5*(-0.0022+2*3.8*10**(-7)*P)
    return tm1, dtm1dp
# melting line 2 for 14.2 wt.% S
def Tm2(P):
    tm2 = 1860.2*(1-0.00512*P-1.22*10**(-5)*P*P)
    dtm2dp = 1860.2*(-0.00512-2*1.22*10**(-5)*P)
    return tm2, dtm2dp
# melting line 3 for 16.2 wt.% S 
def Tm3(P):
    # 23GPa-1333K; 40GPa-1553K
    dtm3dp = (1553-1333)/(40-23)
    tm3 = 1333+(P-23)*dtm3dp
    return tm3, dtm3dp 

#%%
def Rs1(Tcmb):
    # 查找关键Tcmb,当T>Tcmb，绝热线和melting line无交点；当T<Tcmb且Tad>Tl(Rcen)，绝热线和melting line有交点
    # 返回Rs、Rs处温度、RS处压强
    rr = np.linspace(0,R_c,100) #
    TP = np.zeros((len(rr),2))
    for i in range(len(rr)):
        TP[i,:] = TP_calc(rr[i],Tcmb)
    TP1 = TP[np.lexsort(TP[:,::-1].T)] # 按第一列，从小到大排列
    if Tcmb < Tm1(P_cmb0)[0] and TP[0,1] > Tm1(TP[0,0])[0]:  
        rrs = cross_roots(rr,TP[:,1],Tm1(TP[:,0])[:][0]) # rs, Ta, 
        pp = interaction(TP1[:,0],TP1[:,1],rrs[1]*np.ones(len(rr)))
        mm = [rrs[0][0],rrs[1][0],pp[0]]
    else:  
        mm = [R_c,Tcmb,P_cmb0]
    return mm

def Rs2(Tcmb):
    # 查找关键Tcmb,当T>Tcmb，绝热线和melting line无交点；当T<Tcmb且Tad>Tl(Rcen)，绝热线和melting line有交点
    # 返回Rs、Rs处温度、RS处压强
    rr = np.linspace(0,R_c,100) #
    TP = np.zeros((len(rr),2))
    for i in range(len(rr)):
        TP[i,:] = TP_calc(rr[i],Tcmb)
    TP1 = TP[np.lexsort(TP[:,::-1].T)] # 按第一列，从小到大排列
    if Tcmb < Tm2(P_cmb0)[0] and TP[0,1] > Tm2(TP[0,0])[0]:  
        rrs = cross_roots(rr,TP[:,1],Tm2(TP[:,0])[:][0]) # rs, Ta, 
        pp = interaction(TP1[:,0],TP1[:,1],rrs[1]*np.ones(len(rr)))
        mm = [rrs[0][0],rrs[1][0],pp[0]]
    else:  
        mm = [R_c,Tcmb,P_cmb0]
    return mm

def Rs3(Tcmb):
    # 查找关键Tcmb,当T>Tcmb，绝热线和melting line无交点；当T<Tcmb且Tad>Tl(Rcen)，绝热线和melting line有交点
    # 返回Rs、Rs处温度、RS处压强
    rr = np.linspace(0,R_c,100) #
    TP = np.zeros((len(rr),2))
    for i in range(len(rr)):
        TP[i,:] = TP_calc(rr[i],Tcmb)
    TP1 = TP[np.lexsort(TP[:,::-1].T)] # 按第一列，从小到大排列
    if Tcmb < Tm3(P_cmb0)[0] and TP[0,1] > Tm2(TP[0,0])[0]:  
        rrs = cross_roots(rr,TP[:,1],Tm2(TP[:,0])[:][0]) # rs, Ta, 
        pp = interaction(TP1[:,0],TP1[:,1],rrs[1]*np.ones(len(rr)))
        mm = [rrs[0][0],rrs[1][0],pp[0]]
    else:  
        mm = [R_c,Tcmb,P_cmb0]
    return mm

#%%
def dTdP(Tcmb,P_rs):
    # 绝热线斜率dT/dP
    rr = np.linspace(0,R_c,50) #
    TP = np.zeros((len(rr),2))
    for i in range(len(rr)):
        TP[i,:] = TP_calc(rr[i],Tcmb)
    f1 = interp1d(TP[:,0],TP[:,1],kind='quadratic',fill_value="extrapolate")
    dtdp = derivative(f1,P_rs,dx=1e-10) # K/GPa
    return dtdp

#%%

#def Qcmb(x,popt=[1.35334088, 0.69057127, 0.25839456]):
    # 调用 Qc(t,*popt), present is t=0
#    [a0,a1,a2]=popt
#    return 3*a0*np.exp(-(x+4.5)/a1)+a2

#%%
def Qcmb(x):
    return q_cmb0*np.exp(-(x+4.5)/0.69)

#%% Qs
def QQs(Tcmb):
    # dQs / (dTcmb/dt')
    def tr(r):
        return r*r*TP_calc(r,Tcmb)[1]    
    ss, err = integrate.quad(tr,0,R_c) # km^3 K
    return 4*np.pi*rho_c*Cp* ss/Tcmb /(3.1536*10**19) # 10**9 J/K

def EEs(Tcmb):
    # dEs / (dTcmb/dt')
    def tr(r):
        return r*r*TP_calc(r,Tcmb)[1] 
    ss, err =  integrate.quad(tr,0,R_c) # km^3 K
    ees = -4*np.pi*rho_c*Cp*(R_c**3/3/Tcmb- ss/Tcmb/Tcmb) #  
    return ees*10**(-19)/3.5136

#%%
def QQg(Tcmb,index=1):
    # dQg/ (dTcmb/dt')
    # index: 1-Tm1; 2-Tm2; 3-Tm3
    if index==1:
        rs, T_rs, P_rs = Rs1(Tcmb)  # T(rs)/(dTldP-dTdP)
        Pr = 2*np.pi*G/3*rho_c *(R_c*R_c*(1-3*R_c*R_c/10/L/L)-rs*rs*(1-3*rs*rs/10/L/L))
        qqg = -4*np.pi*rs**2*Pr*e_s*T_rs/(dTdP(Tcmb,P_rs)-Tm1(P_rs)[1])/g_calc(rs)/Tcmb  
    elif index==2:
        rs, T_rs, P_rs = Rs2(Tcmb)
        Pr = 2*np.pi*G/3*rho_c *(R_c*R_c*(1-3*R_c*R_c/10/L/L)-rs*rs*(1-3*rs*rs/10/L/L))
        qqg = -4*np.pi*rs**2*Pr*e_s*T_rs/(dTdP(Tcmb,P_rs)-Tm2(P_rs)[1])/g_calc(rs)/Tcmb 
    else:
        rs, T_rs, P_rs = Rs3(Tcmb)
        qqg = 0  
    return -qqg*10**(-7)/3.1536   # TW/(K/Ga)

#%%
def QQL(Tcmb,index=1):
    # dQL / (dTcmb/dt')
    if index==1:
        rs, T_rs, P_rs = Rs1(Tcmb)  # T(rs)/(dTldP-dTdP)
        ds = (1.99731 - 0.0082*P_rs) #*1.38*10**(-23) # J K-1
        qql = 4*np.pi*rs*rs*e_s*T_rs*T_rs*ds/((dTdP(Tcmb,P_rs)-Tm1(P_rs)[1])*Tcmb*g_calc(rs)) 
    elif index==2:
        rs, T_rs, P_rs = Rs2(Tcmb)
        ds = (1.99731 - 0.0082*P_rs) #*1.38*10**(-23) # J K-1
        qql = 4*np.pi*rs*rs*e_s*T_rs*T_rs*ds/((dTdP(Tcmb,P_rs)-Tm2(P_rs)[1])*Tcmb*g_calc(rs)) 
    else:
        rs, T_rs, P_rs = Rs3(Tcmb)
        qql = 0 
    return qql*10**(-13)/3.1536

#%%
def Ek(Tcmb):
    def tr1(r):
        P, T = TP_calc(r, Tcmb)
        V = solve_V(T, P, P_param)
        g= g_calc(r)
        dTdr = rho_c*g*gamma(V,T,P_param,E_param)*T*10**(-6)/Ks(V,T,P_param,E_param)
        return 4*k*np.pi*r*r*(dTdr/T)**2
    ss, err = integrate.quad(tr1,0,R_c) # km^3 K
    return ss*10**(-9) # TW/K




