# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:15:37 2023

@author: lwj
"""
import numpy as np
import states as st
from scipy import integrate

dt = 0.01

T_cmb0 = st.T_cmb0
Ts0 = st.Tm1(st.P_cmb0)[0] # 开始出现Rs的温度
#%% Rs产生后，Tcmb演化规律
def Tcmb_diff1(t,Tcmb):
    dTcmbdt = st.Qcmb(-t)/(st.QQs(Tcmb)-st.QQg(Tcmb)+st.QQL(Tcmb))
    return dTcmbdt

Tcmb_t1 = integrate.solve_ivp(Tcmb_diff1,(0,4.5),[T_cmb0],method='Radau',t_eval=np.arange(0,4.5,dt))

#%%
tts = st.interaction(Tcmb_t1.t, Tcmb_t1.y[0], Ts0*np.ones(len(Tcmb_t1.t))) # 开始出现Rs的时刻

#%% Rs产生前，Tcmb演化规律
def Tcmb_diff2(t,Tcmb):
    # secular cooling 
    dTcmbdt = st.Qcmb(-t)/st.QQs(Tcmb)
    return dTcmbdt

Tcmb_t2 = integrate.solve_ivp(Tcmb_diff2,(tts,4.5),[Ts0],method='Radau',t_eval=np.arange(tts,4.5,dt))

#%% 将Tcmb_t1和Tcmb_t2中满足条件的时间和温度组合
t1 = Tcmb_t1.t[Tcmb_t1.t<tts]
T1 = Tcmb_t1.y[0][:len(t1)]
t_tot = np.concatenate((t1,Tcmb_t2.t),axis=0)
T_tot = np.concatenate((T1, Tcmb_t2.y[0]),axis=0)

#%%
QQ = np.zeros((len(T_tot),12))

#%%
for i in range(len(T_tot)):
    QQ[i,0]=t_tot[i]
    QQ[i,1]=T_tot[i]
    if i <=len(t1):
        QQ[i,2] = Tcmb_diff1(QQ[i,0],QQ[i,1]) # dTcmb/dt
        QQ[i,4] = st.QQg(QQ[i,1])*QQ[i,2]     # Qg
        QQ[i,5] = st.QQL(QQ[i,1])*QQ[i,2]     # QL
        QQ[i,7] = QQ[i,4]/QQ[i,1]             # Eg
        rst, Ts, Ps = st.Rs1(QQ[i,1])
        QQ[i,8] = -QQ[i,5]*(QQ[i,1]-Ts)/Ts/QQ[i,1] # EL
        QQ[i,11] = rst
    else:
        QQ[i,2] = Tcmb_diff2(QQ[i,0],QQ[i,1])
        QQ[i,11] = st.R_c
    QQ[i,3]=st.QQs(QQ[i,1])*QQ[i,2]           # Qs
    QQ[i,6]=st.EEs(QQ[i,1])*QQ[i,2]           # Es
    QQ[i,9]=st.Ek(QQ[i,1])                    # Ek
    QQ[i,10]=(QQ[i,6]+QQ[i,7]+QQ[i,8]-QQ[i,9])*10**6  # EJ, MW/K

#%%
#   import matplotlib.pyplot as plt
#plt.plot(-QQ[:,0],2*QQ[:,4])
# plt.plot(-QQ[:,0],QQ[:,10])

#%%
str1 = "parameter/3Q_Tm1_"+str(st.T_cmb0)+"_"+str(st.P_cmb0)+"_"+str(st.R_c)+"_" \
    +str(st.rho_c)+"_"+str(st.q_cmb0)+"_"+str(st.k)+"_"+str(st.Cp)+'.txt'

#%%
# t, Tcmb_t, dTcmb/dt, Qs, Qg, QL, Es, Eg, EL, Ek, EJ
np.savetxt(str1,QQ,fmt='%.3e',delimiter=" ") 
print(str1)
#%%
# 显示EJ消失时间和Rs出现时间
te = st.cross_roots(QQ[:,0],QQ[:,10],np.zeros(len(QQ)))
#tr = st.cross_roots(QQ[:,0],QQ[:,1],Ts0*np.ones(len(QQ)))
print(4.5-max(te[0]),1830-QQ[0,11])


