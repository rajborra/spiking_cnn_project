from brian2 import *
import numpy as np
import random
defaultclock.dt = 0.01*ms

## Parameters
num_neurons = 7
tau_RC = 10*ms

## Equations
eqs = '''
dv/dt = (-(v+60) + J)/tau_RC:  1
J : 1
'''

## Groups
neurons = NeuronGroup(num_neurons, eqs, clock = Clock(defaultclock.dt), threshold= 'v > -45', reset='v = -60', method='euler')
neurons.v = -60
neurons.J = 0
monitor_v = StateMonitor(neurons,'v',record=True)
monitor_J = StateMonitor(neurons,'J',record=True)

## Synapses
layer1 = [0,1]
layer2 = [2,3,4,5]
layer3 = [6]
w = [random.random()-0.5 for _ in range(8)]

syns = Synapses(neurons, neurons, clock = Clock(defaultclock.dt), model = 'weight:1', on_pre = 'J += weight')
syns.connect(i=[0,0,0,0,1,1,1,1],j=[2,3,4,5,2,3,4,5])
syns.weight = w

## Simulation
neurons.J[0] = 0
run(5*ms)
neurons.J[0] = 50
run(25*ms)
neurons.J[0] = 50
w[3] = -1*w[3]
syns.weight = w
run(50*ms)

## Plotting
figure(1)
subplot(6,1,1)
plot(monitor_v.t/ms, monitor_v.v[0])
subplot(6,1,2)
plot(monitor_v.t/ms, monitor_v.v[1])
subplot(6,1,3)
plot(monitor_J.t/ms, monitor_J.J[2])
subplot(6,1,4)
plot(monitor_J.t/ms, monitor_J.J[3])
subplot(6,1,5)
plot(monitor_J.t/ms, monitor_J.J[4])
subplot(6,1,6)
plot(monitor_J.t/ms, monitor_J.J[5])
xlabel('Time (ms)')
xlim(0,50)
show()
