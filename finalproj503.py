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
# monitr_spike = SpikeMonitor(neurons,,
monitor_J = StateMonitor(neurons,'J',record=True)

## Synapse Initialization
layer1 = [0,1]
layer2 = [2,3,4,5]
layer3 = [6]
w = [random.random()-0.5 for _ in range(12)]

syns = Synapses(neurons, neurons, clock = Clock(defaultclock.dt), model = 'weight:1', on_pre = 'J += 25*weight')
syns.connect(i=[0,0,0,0,1,1,1,1,2,3,4,5],j=[2,3,4,5,2,3,4,5,6,6,6,6])
syns.weight = w

## Simulation
num_trials = 5
data_set = [[0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]]

neurons.J[0] = 0
run(5*ms)
for _ in range(num_trials):
    for set in data_set:
        # Set Spiking Rate for Input Neurons
        neurons.J[0] = set[0]*50
        neurons.J[1] = set[1]*50
        run(25*ms)

        # Calculate Errors
        # err = 

        # Reset Current to Input Neurons
        neurons.J[0] = 0
        neurons.J[1] = 0
        run(5*ms)

## Plotting
figure(1)
subplot(7,1,1)
plot(monitor_v.t/ms, monitor_v.v[0])
subplot(7,1,2)
plot(monitor_v.t/ms, monitor_v.v[1])
subplot(7,1,3)
plot(monitor_J.t/ms, monitor_J.J[2])
subplot(7,1,4)
plot(monitor_J.t/ms, monitor_J.J[3])
subplot(7,1,5)
plot(monitor_J.t/ms, monitor_J.J[4])
subplot(7,1,6)
plot(monitor_J.t/ms, monitor_J.J[5])
subplot(7,1,7)
plot(monitor_v.t/ms, monitor_v.v[6])
xlabel('Time (ms)')
# xlim(0,50)
show()
