from brian2 import *
import numpy as np
import random
import csv
datastruct = []
with open("fisher.csv", "rb") as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:
        templist = []
        templist.append(float(row[0]))
        templist.append(float(row[1]))
        templist.append(float(row[2]))
        templist.append(float(row[3]))
        if(row[4] == "Iris-setosa"):
            templist.append([1,0,0])
        elif(row[4] == "Iris-versicolor"):
            templist.append([0,1,0])
        else:
            templist.append([0,0,1])
        datastruct.append(templist)

defaultclock.dt = 0.01*ms
alpha = 1.0
## CSV


## Parameters
num_neurons = 11
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
#layer1 = [0,1]
#layer2 = [2,3,4,5]
#layer3 = [6]
layer1 = [0,1,2,3]
layer2 = [4,5,6,7]
layer3 = [8,9,10]
w1 = [100*(random.random()-0.5) for _ in range(16)]
w2 = [10*(random.random()-0.5) for _ in range(12)]
#w1 = [-51.763530770485254, -54.332018777533484, 9.4673678308465, -40.17467191494095,
#      14.986568839875105, -10.912079937235054, -39.69347339010056, 17.146279641109103]
#w2 = [2.9015566147797003, 3.8654417015484075, 1.3572763689148397, 2.9648760390418367]
syns = Synapses(neurons, neurons, clock = Clock(defaultclock.dt), 
    model = 'weight:1', on_pre = 'J += weight')
syns.connect(i=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],j=[4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7])
syns.weight = w1

syns2 = Synapses(neurons, neurons, clock = Clock(defaultclock.dt),
 model = 'weight:1', on_pre = 'J += weight')
syns2.connect(i=[4,4,4,5,5,5,6,6,6,7,7,7], j=[8,9,10,8,9,10,8,9,10,8,9,10,])
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
prevcount = [0]*11
epocherr = []
direction = -1
for rand in range(10):
    #
    # if rand > 5 and rand < 10:
    #     alpha = 0.5
    # if rand >= 10:
    #     alpha = 0.25
    temperr = 0
    icount = 1
    print "====="
    print w1
    print w2
    for par in datastruct:
        if rand == 2:
            dire = epocherr[0] - epocherr[1]
            if dire < 0:
                direction = -1
        input = par[0:4]
        expected = par[4]
        neurons.J[0] = 0
        neurons.J[1] = 0
        neurons.J[2] = 0
        neurons.J[3] = 0
        neurons.J[4] = 0
        neurons.J[5] = 0
        neurons.J[6] = 0
        neurons.J[7] = 0
        neurons.J[8] = 0
        neurons.J[9] = 0
        neurons.J[10] = 0
        run(50*ms)
        neurons.J[0] = 10*input[0]
        neurons.J[1] = 10*input[1]
        neurons.J[2] = 10*input[2]
        neurons.J[3] = 10*input[3]
        neurons.J[4] = 0
        neurons.J[5] = 0
        neurons.J[6] = 0
        neurons.J[7] = 0
        neurons.J[8] = 0
        neurons.J[9] = 0
        neurons.J[10] = 0
        run(25*ms)
        neurons.J[0] = 0
        neurons.J[1]= 0
        neurons.J[2] = 0
        neurons.J[3] = 0
        neurons.J[4] = 0
        neurons.J[5] = 0
        neurons.J[6] = 0
        neurons.J[7] = 0
        neurons.J[8] = 0
        neurons.J[9] = 0
        neurons.J[10] = 0
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
        ##############################################
        for c in range(3):
            error = expected[c] - spikes[8+c]
            #print error
            #print "___"
            temperr = temperr + error*error
            if True:
                continue
            grad5 = error
            grads = [val*error for val in w2[c*4:c*4+4]]
            w2[c*4:c*4+4] = [v1+ v2 for v1, v2 in zip(w2,
                [val*grad5*alpha for val in spikes[4:8]])]
            w1[0:4] = [v1 + direction*v2 for v1,v2 in zip( w1[0:4],  
                [val*spikes[0]*alpha for val in grads])]
            w1[4:8] = [v1 + direction*v2 for v1,v2 in zip( w1[4:8],  
                [val*spikes[1]*alpha for val in grads])]
            w1[8:12] = [v1 + direction*v2 for v1,v2 in zip( w1[8:12],  
                [val*spikes[2]*alpha for val in grads])]
            w1[12:16] = [v1 + direction*v2 for v1,v2 in zip( w1[12:16],  
                [val*spikes[3]*alpha for val in grads])]
        ###################################################
        syns.weight = w1
        syns2.weight = w2
        #print icount
        icount = icount + 1;
    epocherr.append(temperr)
    print "Epoch Error is"
    print temperr
    print "========"
    # if temperr < 60:
    #     print "GOOD WEIGHTS"
    #     print w1
    #     print w2
    #     break;
    #print w1
    #print w2
print "W1"
print w1
print "w2"
print w2
ferror = 0
for par in []:
    input = par[0:4]
    expected = par[4]
    neurons.J[0] = 0
    neurons.J[1] = 0
    neurons.J[2] = 0
    neurons.J[3] = 0
    neurons.J[4] = 0
    neurons.J[5] = 0
    neurons.J[6] = 0
    neurons.J[7] = 0
    neurons.J[8] = 0
    neurons.J[9] = 0
    neurons.J[10] = 0
    run(50*ms)
    neurons.J[0] = 10*input[0]
    neurons.J[1] = 10*input[1]
    neurons.J[2] = 10*input[2]
    neurons.J[3] = 10*input[3]
    neurons.J[4] = 0
    neurons.J[5] = 0
    neurons.J[6] = 0
    neurons.J[7] = 0
    neurons.J[8] = 0
    neurons.J[9] = 0
    neurons.J[10] = 0
    run(25*ms)
    neurons.J[0] = 0
    neurons.J[1]= 0
    neurons.J[2] = 0
    neurons.J[3] = 0
    neurons.J[4] = 0
    neurons.J[5] = 0
    neurons.J[6] = 0
    neurons.J[7] = 0
    neurons.J[8] = 0
    neurons.J[9] = 0
    neurons.J[10] = 0
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
    ##############################################
    for c in range(3):
        error = expected[c] - spikes[8+c]
        #print error
        #print "___"
        ferror = ferror + error*error
        #grad5 = error
        #grads = [val*error for val in w2[c*4:c*4+4]]
        #w2[c*4:c*4+4] = [v1+ v2 for v1, v2 in 
        #                 zip(w2,[val*grad5*alpha for val in spikes[4:8]])]
        #w1[0:4] = [v1 + direction*v2 for v1,v2 in zip( w1[0:4],  
        #         [val*spikes[0]*alpha for val in grads])]
        #w1[4:8] = [v1 + direction*v2 for v1,v2 in zip( w1[4:8],  
        #         [val*spikes[1]*alpha for val in grads])]
        #w1[8:12] = [v1 + direction*v2 for v1,v2 in zip( w1[8:12],  
        #           [val*spikes[2]*alpha for val in grads])]
        #w1[12:16] = [v1 + direction*v2 for v1,v2 in zip( w1[12:16],  
        #           [val*spikes[3]*alpha for val in grads])]
    ###################################################
    syns.weight = w1
    syns2.weight = w2
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
