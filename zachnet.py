from brian2 import *
import numpy as np
import random
defaultclock.dt = 0.01*ms
alpha = 0.1
## Parameters
num_neurons = 7
tau_RC = 10*ms

## Equations
eqs = '''
dv/dt = (-0.5*(v+60) + J-5)/tau_RC:  1
J : 1
'''

## Groups
neurons = NeuronGroup(num_neurons, eqs, clock = Clock(defaultclock.dt), threshold= 'v > -45', reset='v = -60', method='euler')
neurons.v = -60
neurons.J = 0
monitor_v = StateMonitor(neurons,'v',record=True)
monitor_J = StateMonitor(neurons,'J',record=True)
k = SpikeMonitor(neurons)

## Synapses
layer1 = [0,1]
layer2 = [2,3,4,5]
layer3 = [6]
w1 = [100*(random.random()-0.5) for _ in range(8)]
w2 = [100*(random.random()-0.5) for _ in range(4)]
w1 = [25,25,25,25,25,25,25,25]
w2 = [25,25,25,25]
syns = Synapses(neurons, neurons, clock = Clock(defaultclock.dt), model = 'weight:1', on_pre = 'J += weight')
syns.connect(i=[0,0,0,0,1,1,1,1],j=[2,3,4,5,2,3,4,5])
syns.weight = w1

syns2 = Synapses(neurons, neurons, clock = Clock(defaultclock.dt), model = 'weight:1', on_pre = 'J += weight')
syns2.connect(i=[2,3,4,5], j=[6,6,6,6])
syns2.weight = w2
## Simulation
# print "first"
# print w1
# print w2
# neurons.J[0] = 0
# run(5*ms)
# neurons.J[0] = 50
# run(25*ms)
# input = [1,0]
# expected = 1
# spikes = [0]*7
# max = max(k.count)
# print k.count
# for i in range(7):
    # spikes[i] = (float(k.count[i])/float(max))
# print spikes
# error = expected - spikes[6]
# grad5 = error
# grads = [val*error for val in w2]
# print grads
# w2 = [v1+ v2 for v1, v2 in zip(w2,[val*grad5*alpha for val in spikes[2:6]])]
# w1[0:4] = [v1 + v2 for v1,v2 in zip( w1[0:4],  [val*input[0]*alpha for val in grads])]
# w1[4:8] = [v1 + v2 for v1,v2 in zip( w1[4:8],  [val*input[1]*alpha for val in grads])]
# syns.weight = w1
# syns2.weight = w2
# neurons.J[0] = 0
# run(5*ms)
# neurons.J[0] = 50
# run(25*ms)
#w[3] = -1*w[3]
#syns.weight = w
#print "second"
#print w1
#print w2
#print k.count
print "------------------------------------------"
prevcount = [0]*7
epocherr = []
for rand in range(0):
    temperr = 0
    for par in [[1,1,1],[1,0,1],[0,0,0],[0,1,1]]:
        input = par[0:2]
        expected = par[2]
        neurons.J[0] = 0
        neurons.J[1] = 0
        run(50*ms)
        neurons.J[0] = 50*input[0]
        neurons.J[1] = 50*input[1]
        run(25*ms)
        neurons.J[0] = 0
        neurons.J[1]= 0
        run(50*ms)
        ncount = [v1-v2+1 for v1,v2 in zip(k.count,prevcount)]
        prevcount = k.count[:]
        if max(ncount)==0:
            temp = 1
        else:
            temp = 0
        spikes = [float(val)/(float(max(ncount))+temp) for val in ncount]
        error = expected - spikes[6]
        temperr = temperr + error*error
        grad5 = error
        grads = [val*error for val in w2]
        w2 = [v1+ v2 for v1, v2 in zip(w2,[val*grad5*alpha for val in spikes[2:6]])]
        w1[0:4] = [v1 + v2 for v1,v2 in zip( w1[0:4],  [val*spikes[0]*alpha for val in grads])]
        w1[4:8] = [v1 + v2 for v1,v2 in zip( w1[4:8],  [val*spikes[1]*alpha for val in grads])]
        syns.weight = w1
        syns2.weight = w2
    epocherr.append(temperr)
    #print w1
    #print w2
print "W1"
print w1
print "w2"
print w2
for par in [[0,1,1],[1,0,1],[0,0,0],[1,1,1]]:
    input = par[0:2]
    expected = par[2]
    neurons.J[0] = 0
    neurons.J[1] = 0
    run(5*ms)
    neurons.J[0] = 50*input[0]
    neurons.J[1] = 50*input[1]
    run(25*ms)
    neurons.J[0] = 0
    neurons.J[1]= 0
    run(5*ms)
    ncount = [v1-v2 for v1,v2 in zip(k.count,prevcount)]
    if max(ncount)==0:
        temp = 1
    else:
        temp = 0
    prevcount = k.count[:]
    spikes = [float(val)/(float(max(ncount)+temp)) for val in ncount]
    print "spikes"
    print spikes
    #grad5 = error
    #grads = [val*error for val in w2]
    #w2 = [v1+ v2 for v1, v2 in zip(w2,[val*grad5*alpha for val in spikes[2:6]])]
    #w1[0:4] = [v1 + v2 for v1,v2 in zip( w1[0:4],  [val*input[0]*alpha for val in grads])]
    #w1[4:8] = [v1 + v2 for v1,v2 in zip( w1[4:8],  [val*input[1]*alpha for val in grads])]
    #syns.weight = w1
    #syns2.weight = w2
print("ERROR")
print epocherr


# Plotting

figure(1)
subplot(7,1,1)
plot(monitor_v.t/ms, monitor_v.v[0])
subplot(7,1,2)
plot(monitor_v.t/ms, monitor_v.v[1])
subplot(7,1,3)
plot(monitor_v.t/ms, monitor_v.v[2])
subplot(7,1,4)
plot(monitor_v.t/ms, monitor_v.v[3])
subplot(7,1,5)
plot(monitor_v.t/ms, monitor_v.v[4])
subplot(7,1,6)
plot(monitor_v.t/ms, monitor_v.v[5])
subplot(7,1,7)
plot(monitor_v.t/ms, monitor_v.v[6])
xlabel('Time (ms)')
xlim(0,50)
show()
