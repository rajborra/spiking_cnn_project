from brian2 import *
import numpy as np
import random
defaultclock.dt = 0.01*ms
alpha = 1.0
## Parameters
num_neurons = 7
tau_RC = 3*ms

## Equations
eqs = '''
dv/dt = (-0.5*(v+60) + J)/tau_RC:  1
J : 1
'''

learning = [[0,1,1.0],[1,0,1.0],[0,0,0.0],[1,1,0.0]]
## Groups
neurons = NeuronGroup(num_neurons, eqs, clock = Clock(defaultclock.dt), 
    threshold= 'v > -45', reset='v = -60', method='euler')
neurons.v = -60
neurons.J = 0
monitor_v = StateMonitor(neurons,'v',record=True)
monitor_J = StateMonitor(neurons,'J',record=True)
k = SpikeMonitor(neurons)

## Synapses
layer1 = [0,1]
layer2 = [2,3,4,5]
layer3 = [6]
#w1 = [100*(random.random()-0.5) for _ in range(8)]
#w2 = [10*(random.random()-0.5) for _ in range(4)]
#w1 = [25,25,25,25,25,25,25,25]
#w2 = [25,25,25,25]
w1 = [-51.763530770485254, -54.332018777533484, 9.4673678308465, -40.17467191494095, 
      14.986568839875105, -10.912079937235054, -39.69347339010056, 17.146279641109103]
w2 = [2.9015566147797003, 3.8654417015484075, 1.3572763689148397, 2.9648760390418367]
syns = Synapses(neurons, neurons, clock = Clock(defaultclock.dt),
       model = 'weight:1', on_pre = 'J += weight')
syns.connect(i=[0,0,0,0,1,1,1,1],j=[2,3,4,5,2,3,4,5])
syns.weight = w1

syns2 = Synapses(neurons, neurons, clock = Clock(defaultclock.dt),
        model = 'weight:1', on_pre = 'J += weight')
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
# w1[0:4] = [v1 + v2 for v1,v2 in zip( w1[0:4],
#           [val*input[0]*alpha for val in grads])]
# w1[4:8] = [v1 + v2 for v1,v2 in zip( w1[4:8], 
#           [val*input[1]*alpha for val in grads])]
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
direction = -1
for rand in range(7):
    #
    # if rand > 5 and rand < 10:
    #     alpha = 0.5
    # if rand >= 10:
    #     alpha = 0.25
    temperr = 0
    for par in [[0,1,1.0],[1,0,1.0],[0,0,0.0],[1,1,0.0]]:
        if rand == 2:
            dire = epocherr[0] - epocherr[1]
            if dire < 0:
                direction = -1
        input = par[0:2]
        expected = par[2]
        neurons.J[0] = 0
        neurons.J[1] = 0
        neurons.J[2] = 0
        neurons.J[3] = 0
        neurons.J[4] = 0
        neurons.J[5] = 0
        neurons.J[6] = 0
        run(50*ms)
        neurons.J[0] = 50*input[0]
        neurons.J[1] = 50*input[1]
        neurons.J[2] = 0
        neurons.J[3] = 0
        neurons.J[4] = 0
        neurons.J[5] = 0
        neurons.J[6] = 0
        run(25*ms)
        neurons.J[0] = 0
        neurons.J[1]= 0
        neurons.J[2] = 0
        neurons.J[3] = 0
        neurons.J[4] = 0
        neurons.J[5] = 0
        neurons.J[6] = 0
        run(50*ms)
        ncount = [v1-v2 for v1,v2 in zip(k.count,prevcount)]
        prevcount = k.count[:]
        if max(ncount)!=0:
            spikes = [float(val)/(float(max(ncount))) for val in ncount]
        else:
            spikes = ncount[:]
        #spikes = [float(val)/(float(max(ncount))+temp) for val in ncount]
        #print spikes
        #print expected
        #print spikes[6]
        error = expected - spikes[6]
        #print error
        #print "___"
        temperr = temperr + error*error
        grad5 = error
        grads = [val*error for val in w2]
        w2 = [v1+ v2 for v1, v2 in zip(w2,[val*grad5*alpha for val in spikes[2:6]])]
        w1[0:4] = [v1 + direction*v2 for v1,v2 in zip( w1[0:4],  
            [val*spikes[0]*alpha for val in grads])]
        w1[4:8] = [v1 + direction*v2 for v1,v2 in zip( w1[4:8],  
            [val*spikes[1]*alpha for val in grads])]
        syns.weight = w1
        syns2.weight = w2
    epocherr.append(temperr)
    print temperr
    #print w1
    #print w2
print "W1"
print w1
print "w2"
print w2
ferror = 0
for par in [[0,1,1.0],[1,0,1.0],[0,0,0.0],[1,1,0.0]]:
    input = par[0:2]
    expected = par[2]
    neurons.J[0] = 0
    neurons.J[1] = 0
    neurons.J[2] = 0
    neurons.J[3] = 0
    neurons.J[4] = 0
    neurons.J[5] = 0
    neurons.J[6] = 0
    run(50*ms)
    neurons.J[0] = 50*input[0]
    neurons.J[1] = 50*input[1]
    neurons.J[2] = 0
    neurons.J[3] = 0
    neurons.J[4] = 0
    neurons.J[5] = 0
    neurons.J[6] = 0
    run(25*ms)
    neurons.J[0] = 0
    neurons.J[1]= 0
    neurons.J[2] = 0
    neurons.J[3] = 0
    neurons.J[4] = 0
    neurons.J[5] = 0
    neurons.J[6] = 0
    run(50*ms)
    ncount = [v1-v2 for v1,v2 in zip(k.count,prevcount)]
    if max(ncount)!=0:
        spikes = [float(val)/(float(max(ncount))) for val in ncount]
    else:
        spikes = ncount[:]
    prevcount = k.count[:]
    #spikes = [float(val)/(float(max(ncount)+temp)) for val in ncount]
    print "spikes"
    print spikes
    nerr = expected - spikes[6]
    print nerr
    ferror = ferror + nerr*nerr
    #grad5 = error
    #grads = [val*error for val in w2]
    #w2 = [v1+ v2 for v1, v2 in zip(w2,[val*grad5*alpha for val in spikes[2:6]])]
    #w1[0:4] = [v1 + v2 for v1,v2 in zip( w1[0:4],  
    #          [val*input[0]*alpha for val in grads])]
    #w1[4:8] = [v1 + v2 for v1,v2 in zip( w1[4:8],  
    #          [val*input[1]*alpha for val in grads])]
    #syns.weight = w1
    #syns2.weight = w2
print("ERROR")
print epocherr
print("FINAL ERROR")
print ferror


# Plotting

# figure(1)
# subplot(7,1,1)
# plot(monitor_v.t/ms, monitor_v.v[0])
# subplot(7,1,2)
# plot(monitor_v.t/ms, monitor_v.v[1])
# subplot(7,1,3)
# plot(monitor_v.t/ms, monitor_v.v[2])
# subplot(7,1,4)
# plot(monitor_v.t/ms, monitor_v.v[3])
# subplot(7,1,5)
# plot(monitor_v.t/ms, monitor_v.v[4])
# subplot(7,1,6)
# plot(monitor_v.t/ms, monitor_v.v[5])
# subplot(7,1,7)
# plot(monitor_v.t/ms, monitor_v.v[6])
# xlabel('Time (ms)')
# #xlim(0,125)
show()