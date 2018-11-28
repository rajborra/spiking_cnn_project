#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:32:28 2018

@author: chenriq
"""

from brian2 import *
import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot
from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs
from pylab import *
from numpy import *
from brian2 import PopulationRateMonitor
from brian2 import SpikeMonitor
import random
import sys
import select
import os
import time
import math
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel, convolve
import scipy.io as sio
start_scope()


Tmax=200
defaultclock.dt=0.1*ms
dt=defaultclock.dt/ms



# Target Patterns
targOut=[[22.0, 55.0, 72.0, 90.0, 144.0, 180.0],[22.0, 55.0, 90.0, 121.0, 154.0, 187.0],[  22.,  180.,  162.,   90.,   72.,   54.,  144.,   36.],[22.0, 55.0, 90.0, 121.0, 154.0, 187.0],[22.0, 55.0, 90.0, 121.0, 154.0, 187.0]]
N=6
NH=6
N2=1

sd=np.zeros((int(Tmax/dt)+1,N2))
si=np.zeros((int(Tmax/dt)+1,NH))
so=np.zeros((int(Tmax/dt)+1,N2))
sh=np.zeros((int(Tmax/dt)+1,N2))

for j in range(N2):
  for i in range(0,int(Tmax/dt)+1):
     if 0.1*i in targOut[j][:]:
        sd[i,j]=1
     else:
        sd[i,j]=0


A=0.000002


Tw=np.exp(-np.arange(0,11)/5.0)

i1=arange(17.0,200.0,16.0)
i2=arange(10.9,200,11.0)
i3=arange(8.5,200.0,8.2)
i4=arange(17.9,200.0,18.0)
i5=arange(10.0,200,11.0)
i6=arange(8.0,200.0,8.2)




######### Build first neuron group with N neurons ##########
#NR=25
#random.seed(8)
#i1=numpy.unique(numpy.sort(randint(0,200,NR)))*1.0
#i2=numpy.unique(numpy.sort(randint(0,200,NR+1)))*1.0
#i3=numpy.unique(numpy.sort(randint(0,200,NR-3)))*1.0
#i4=numpy.unique(numpy.sort(randint(0,200,NR)))*1.0
#i5=numpy.unique(numpy.sort(randint(0,200,NR-4)))*1.0
#i6=numpy.unique(numpy.sort(randint(0,200,NR-2)))*1.0
#i7=numpy.unique(numpy.sort(randint(0,200,NR)))*1.0
#i8=numpy.unique(numpy.sort(randint(0,200,NR+10)))*1.0
#i9=numpy.unique(numpy.sort(randint(0,200,NR-5)))*1.0
#i10=numpy.unique(numpy.sort(randint(0,200,NR)))*1.0
#i11=numpy.unique(numpy.sort(randint(0,200,NR-4)))*1.0
#i12=numpy.unique(numpy.sort(randint(0,200,NR-20)))*1.0


#itot=numpy.concatenate((i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12), axis=0, out=None)
itot=numpy.concatenate((i1,i2,i3,i4,i5,i6), axis=0, out=None)
input_times=itot*ms
#input_times = array([ 17.9,  35.9,  53.9,  71.9,  89.9, 107.9, 125.9, 143.9, 161.9,
#       179.9, 197.9,
#       10.9,  21.9,  32.9,  43.9,  54.9,  65.9,  76.9,  87.9,  98.9,
#       109.9, 120.9, 131.9, 142.9, 153.9, 164.9, 175.9, 186.9, 197.9,
#       8. ,  16.2,  24.4,  32.6,  40.8,  49. ,  57.2,  65.4,  73.6,
#        81.8,  90. ,  98.2, 106.4, 114.6, 122.8, 131. , 139.2, 147.4,
#       155.6, 163.8, 172. , 180.2, 188.4, 196.6])*ms 

a1=repeat(0,len(i1),axis=0) 
a2=repeat(1,len(i2),axis=0) 
a3=repeat(2,len(i3),axis=0) 
a4=repeat(3,len(i4),axis=0) 
a5=repeat(4,len(i5),axis=0) 
a6=repeat(5,len(i6),axis=0)  
#a7=repeat(6,len(i7),axis=0) 
#a8=repeat(7,len(i8),axis=0) 
#a9=repeat(8,len(i9),axis=0) 
#a10=repeat(9,len(i10),axis=0) 
#a11=repeat(10,len(i11),axis=0) 
#a12=repeat(11,len(i12),axis=0)  
# input_indices = numpy.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12), axis=0, out=None)
input_indices = numpy.concatenate((a1,a2,a3,a4,a5,a6), axis=0, out=None)   
#                       
#                       

tau = 10*ms
Er=-65*mV
R=10000000*ohm

eqs='''
dv/dt = (-(v-Er)+R*Iinj)/tau : volt
Iinj:amp
'''

G= SpikeGeneratorGroup(N, input_indices, input_times)

#G = NeuronGroup(N, eqs, clock=Clock(defaultclock.dt),threshold='v>-55*mV', reset='v=Er',refractory=4*ms,
#                method='linear')
########## Build second neuron group with N2 neurons ##########


H=NeuronGroup(NH, eqs, clock=Clock(defaultclock.dt),threshold='v>-55*mV', reset='v=Er',refractory=4*ms,
                method='linear')


T = NeuronGroup(N2, eqs, clock=Clock(defaultclock.dt),threshold='v>-55*mV', reset='v=Er',refractory=4*ms,
                method='linear')

#G.v='Er'
T.v='Er'
H.v='Er'


SIH = Synapses(G, H, 'w1: 1', on_pre='v_post+=10**9*w1*mV')
#SIH.connect(True)
SIH.connect(i=[0,1,2,3,4,5],j=[0,1,2,3,4,5])
#SIH.connect(i=[0,1,2,3,4,5,6,7,8,9],j=[0,1,2,3,4,5,6,7,8,9]) #all to all connections
#gmax =.0000075+.0000075*np.random.normal(0,0.4,len(S.w))

######### connect first group to second group #############
S = Synapses(H, T, 'w: 1', on_pre='v_post+=10**9*w*mV')
S.connect(True) 
 #all to all connections
#gmax =.0000075+.0000075*np.random.normal(0,0.4,len(S.w))



gmax=7.5*10**-6
print(gmax)

#S.w[:,:]= gmax+gmax*np.random.normal(0,0.4,len(S.w))
S.w[:,:]= gmax
#+gmax*np.random.normal(0,0.4,len(S.w))
SIH.w1[:,:]=gmax

#gmax*np.random.normal(0,0.4,len(SIH.w1))
#numpy.random.seed(6);SIH.delay=np.random.uniform(low=1, high=5, size=len(SIH.w1))*ms

#SIH.delay=1*ms


lr=gmax/90

print(S.w)
print(SIH.w1)
######## define the inputs to the first group #############
def error(spikemonout,sd,pattern,so):
    spike_out=spikemonout.spike_trains()
    #print(spike_out[pattern])

    for ti in spike_out[pattern]:        
       so[int(float(ti)*1000/0.1)]=1
       
    if len(spike_out[pattern])==0:
       return 0  
   # Create kernel
    g = np.sqrt(2*3.1415926)*4*Gaussian1DKernel(stddev=4)

   # Convolve data
    vd = convolve(sd, g)
       
    vo = convolve(so, g)
  
    C  = np.dot(vd,vo)/(np.linalg.norm(vd)*np.linalg.norm(vo))
    
    print 'return cost for pattern ', pattern+1
    print(C)

   
    return C

#M = StateMonitor(G, 'v', record=True)
spikemon = SpikeMonitor(G)
spikemonhidden=SpikeMonitor(H)
spikemonout = SpikeMonitor(T)

#stimulus into input layer
numpy.random.seed(6);inputcase=0.4*10**-9*np.random.randint(1, 6+1, N)*amp;

#numpy.random.seed(6);inputcase=[4*0.4*10**-9,6*0.4*10**-9] *amp;

P=0.01

iters=900 #number of max iterations
dwdt=np.zeros((NH,N2))
dIHwdt=np.zeros((N,NH))
#print(S.w) 

#used to store the state of the system
#store(filename='stored_state.pickle')
store()
stored_w = Quantity(S.w, copy=True)
stored_IHw = Quantity(SIH.w1, copy=True)

P_total=np.zeros((N2,iters)) 
initial=1
final=0
Weighttotal=[]
Weighttotal2=[]
testindex=0

while min(P_total[:,testindex-1])<0.95:  
#  print(inputcase)
#  G.Iinj=inputcase
  S.w[:,:] = stored_w 
  SIH.w1[:,:] = stored_IHw 
  Weighttotal.append(array(S.w)) 
  Weighttotal2.append(array(SIH.w1)) 
  run(Tmax*ms)
  if initial==1:
      spikeinitial_i=array(spikemon.i)
      spikeinitial_t=array(spikemon.t)
      spikeinitial_h=array(spikemonhidden.i)
      spikeinitial_h_t=array(spikemonhidden.t)
      spikeinitial_out=array(spikemonout.i)
      spikeinitial_out_t=array(spikemonout.t)  
      initial=0
    
  for pattern in range(N2):
      
      P=error(spikemonout,sd[:,pattern],pattern,so[:,pattern])  
      if P>0.95:
          spikefinal_out=array(spikemonout.i)
          spikefinal_out_t=array(spikemonout.t) 
      
      P_total[pattern,testindex]=P  
#input spikes into bit stream      
      for i in range(NH):
        
        
           

        
        spike_hidden=spikemonhidden.spike_trains()
        sh=np.zeros((int(Tmax/dt)+1,N2))
        for ti in spike_hidden[i]:
           sh[int(float(ti)*1000/0.1),pattern]=1

        spike_trains_out=spikemonout.spike_trains() 
        so=np.zeros((int(Tmax/dt)+1,N2))
        for ti in spike_trains_out[pattern]:
           so[int(float(ti)*1000/0.1),pattern]=1
        
        conv=convolve(A*sh[:,pattern],Tw) 
        dwdt[i,pattern]=sum((1.0/NH)*(lr*(sd[:,pattern]-so[:,pattern])+(sd[:,pattern]-so[:,pattern])*conv))
        S.w[i,pattern]=S.w[i,pattern]+dwdt[i,pattern]
        
        rhidden=sum(sh[:,pattern])/(Tmax*ms/second)
#        print pattern, rhidden
        #SIH.w1[i,kk]=SIH.w1[i,kk]   
        
        
        
  print 'update ', testindex, ' finished'
  figure()
  plot(spikemonout.t/ms,spikemonout.i,'k*')
  plot(targOut[0],[0.2, 0.2, 0.2, 0.2, 0.2, 0.2],'r*')
  ylim(-.3,2)
#  figure()
#  plot(spikemon.t/ms,spikemon.i,'k*')
#  figure()
#  plot(spikemonhidden.t/ms,spikemonhidden.i,'k*') 
  
#  plot(targOut[1],[1.2, 1.2,1.2, 1.2,1.2, 1.2],'r*') 
#  plot(targOut[2],[2.2, 2.2,2.2, 2.2,2.2, 2.2,2.2,2.2],'r*') 
#  plot(targOut[3],[3.2, 3.2,3.2, 3.2,3.2, 3.2],'r*') 
#  plot(targOut[4],[4.2, 4.2,4.2, 4.2,4.2, 4.2],'r*')  
  show() 
  stored_w = Quantity(S.w, copy=True)
  stored_IHw = Quantity(SIH.w1, copy=True)
  testindex=testindex+1      
  #print(S.w)   
  #print(dwdt)  
#  print(spikemon.t)
  #print(spikemonout.t)
#  print(P)  
  
  
  #restore(filename='stored_state.pickle')
  restore()


plot(Weighttotal)    
    
    
sio.savemat('outputsaved',{'Weighttotal':Weighttotal,'spikeinitial_out_t':spikeinitial_out_t,'spikeinitial_out':spikeinitial_out,'spikefinal_out_t':spikefinal_out_t,
           'spikefinal_out':spikefinal_out,'P_total':P_total})