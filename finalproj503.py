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
monitor_s = SpikeMonitor(neurons)
monitor_J = StateMonitor(neurons,'J',record=True)

## Synapse Initialization
layer1 = [0,1]
layer2 = [2,3,4,5]
layer3 = [6]
w = [random.random()-0.5 for _ in range(12)]

syns = Synapses(neurons, neurons, clock = Clock(defaultclock.dt), model = 'weight:1', on_pre = 'J += 1000*weight')
syns.connect(i=[0,0,0,0,1,1,1,1,2,3,4,5],j=[2,3,4,5,2,3,4,5,6,6,6,6])
syns.weight = w

## Simulation
alpha = 0.1
num_trials = 10
data_set = [[0, 0, 0],                              # [0,1] are inputs, [2] is expected output
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]]

neurons.J[0] = 0
run(5*ms)

spike_count_current = monitor_s.count
curr_time = 0

counter = 0;

# Iterate through Trials
for _ in range(num_trials):
    # Iterate through Dataset
    for set in data_set:
        # Set Spiking Rate for Input Neurons
        neurons.J[0] = set[0]*50                    # set[0] = first value in each set in dataset
        neurons.J[1] = set[1]*50                    # set[1] = second value in each set in dataset
        expected = set[2]
        run(25*ms)

        v_avg = [0 for _ in range(num_neurons)]
        for i in range(num_neurons):
            v_avg[i] = sum(monitor_v.v[i][curr_time:curr_time+2500])/2500 + 60

        # # Calculate Number of Spikes at Output
        # spike_count_current = monitor_s.count - spike_count_current
        # has_spiked = [0 for _ in range(len(spike_count_current))]
        # for i in range(len(spike_count_current)):
        #     if spike_count_current[6] > 0:
        #         has_spiked[i] = 1
        #     else:
        #         has_spiked[i] = 0
        # actual = has_spiked[6]

        # Calculate Error


        err = [0 for _ in range(num_neurons)]
        if 6*expected-v_avg[6] < 1 and 6*expected-v_avg[6] > -1:
            diff = 0
        else:
            counter += 1
            diff = 6*expected-v_avg[6]
        err[6] = (diff)*v_avg[6]
        for i in range(num_neurons):
            if i in layer1:
                err[i] = 0
            if i in layer2:
                err[i] = sum([err[6]*val for val in w[8:]])*v_avg[i]

        # Update New Weights

        index = 0
        for i in layer1:
            for j in layer2:
                w[index] += -alpha*err[j]*v_avg[i]
                index += 1

        for i in layer2:
            for j in layer3:
                w[index] += -alpha*err[j]*v_avg[i]
                index += 1

        syns.weight = w

        # Reset Current to Input Neurons
        neurons.J[0] = 0
        neurons.J[1] = 0
        run(5*ms)
        curr_time += 3000

print counter
## Plotting
figure(1)
subplot(8,1,1)
plot(monitor_v.t/ms, monitor_v.v[0])
subplot(8,1,2)
plot(monitor_v.t/ms, monitor_v.v[1])
subplot(8,1,3)
plot(monitor_J.t/ms, monitor_J.J[2])
subplot(8,1,4)
plot(monitor_J.t/ms, monitor_J.J[3])
subplot(8,1,5)
plot(monitor_J.t/ms, monitor_J.J[4])
subplot(8,1,6)
plot(monitor_J.t/ms, monitor_J.J[5])
subplot(8,1,7)
plot(monitor_J.t/ms, monitor_J.J[6])
subplot(8,1,8)
plot(monitor_v.t/ms, monitor_v.v[6])
xlabel('Time (ms)')
# xlim(0,50)
show()
