from brian2 import *
import numpy as np
import random
import math
dt = 0.1*ms; defaultclock.dt = dt;
trial_duration = 50*ms

## Parameters
mem_rest = 0
tau = 5*ms

## Equations
eqs = '''
dv/dt = (-v+g)/tau : 1
dg/dt = -g/tau : 1
'''

## Groups
num_inputs = 3
inputs = SpikeGeneratorGroup(3, [0,1,2],[0,0,0]*ms)

num_hidden = 4
hidden = NeuronGroup(num_hidden, eqs, clock = Clock(defaultclock.dt), 
    threshold= 'v > 5', reset='v = mem_rest', method='euler')

output = NeuronGroup(1, eqs, clock = Clock(defaultclock.dt), 
    threshold= 'v > 5', reset='v = mem_rest', method='euler')

hidden.v = mem_rest; hidden.g = 0
output.v = mem_rest; output.g = 0

## Synapse Initialization

possible_weights = [0, 0.5, 1, 1.5]

delay_range = range(16)
inp_hidden = [0 for _ in delay_range]
hidden_out = [0 for _ in delay_range]

inp_hidden = Synapses(inputs, hidden, clock = Clock(defaultclock.dt), model = 'w:1 ', 
    on_pre = 'g += w', multisynaptic_index='syn_num')
hidden_out = Synapses(hidden, output, clock = Clock(defaultclock.dt), model = 'w:1 ', 
    on_pre = 'g += w', multisynaptic_index='syn_num')

inp_hidden.connect(i=[0,0,0,0,1,1,1,1,2,2,2,2],j=[0,1,2,3,0,1,2,3,0,1,2,3],n=16)
hidden_out.connect(i=[0,1,2,3],j=[0,0,0,0],n=16)

inp_hidden.w = 0.2
hidden_out.w = 0.2

inp_hidden.delay = '1*ms+syn_num*ms'
hidden_out.delay = '1*ms+syn_num*ms'

# Simulation

monitor_input = SpikeMonitor(inputs,record=True)
monitor_input_spikes = SpikeMonitor(inputs,record=True)
monitor_hidden = StateMonitor(hidden,('v','g'),record=True)
monitor_output = StateMonitor(output,('v','g'),record=True)

num_trials = 500

data_set = [[0, 0, 0],               # [0,1] are inputs, [2] is expected output
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]]

curr_time = 0
convergence = 5
error_val = float("inf")

for trial in range(num_trials):
    print "Trial", trial+1
    errors = []
    for set_num in range(len(data_set)):
        monitor_output_spikes = SpikeMonitor(output,('v','g'),record=True)
        set = data_set[set_num]
        input_spike_times = [6*time+curr_time for time in [0]+set[0:2]]*ms
        inputs.set_spikes([0,1,2],input_spike_times)
        run(trial_duration)

        expected_output = (16 - set[2]*6) + curr_time

        if len(monitor_output_spikes.t) == 0:
            actual_output = 20
        else:
            actual_output = monitor_output_spikes.t[0]/ms

        times = [j for j in monitor_output.t/ms]


        diff_output = actual_output - expected_output
        print "Set", set_num, "Error :", diff_output
        errors.append(diff_output)

        for delay_val in range(16):
            spike_index = times.index(actual_output)-delay_val*10
            sum_vals = 0
            g_vals = []spi

            ## Hidden to Output Layer
            for hidden_neuron in range(num_hidden):
                current_weight = hidden_out.w[delay_val+hidden_neuron*16]
                g_val = monitor_hidden.g[hidden_neuron][spike_index]
                g_vals.append(g_val)
                sum_vals += -g_val/tau
            for hidden_neuron in range(num_hidden):
                current_weight = hidden_out.w[delay_val+hidden_neuron*16]
                current_weight += 0.001*g_vals[hidden_neuron]*diff_output/current_weight
                hidden_out.w[delay_val+hidden_neuron*16] = min(current_weight,1.5)

            ## Input to Hidden Layer
            for input_neuron in range(num_inputs):
                for hidden_neuron in range(num_hidden):
                    current_weight = 
                          inp_hidden.w[delay_val+(input_neuron*4+hidden_neuron)*16]
                    current_weight += 0.001*g_vals[hidden_neuron]*diff_output/current_weight
                    inp_hidden.w[delay_val+(input_neuron*4+hidden_neuron)*16] = 
                           min(current_weight,1.5)


        curr_time += trial_duration/ms
    mse = sum([error**2 for error in errors])/2
    print mse
    if mse < 8:
        break

# Plotting
# figure(1)
# subplot(8,1,1)
# plot(monitor_input_spikes.t[0])
# subplot(8,1,2)
# plot(monitor_input_spikes.t[1])
# subplot(8,1,3)
# plot(monitor_input_spikes.t[2])
# subplot(8,1,4)
# plot(monitor_hidden.t/ms, monitor_hidden.v[0])
# subplot(8,1,5)
# plot(monitor_hidden.t/ms, monitor_hidden.v[1])
# subplot(8,1,6)
# plot(monitor_hidden.t/ms, monitor_hidden.v[2])
# subplot(8,1,7)
# plot(monitor_hidden.t/ms, monitor_hidden.v[3])
# subplot(8,1,8)
# plot(monitor_output.t/ms, monitor_output.v[0])
# xlabel('Time (ms)')
# show()
