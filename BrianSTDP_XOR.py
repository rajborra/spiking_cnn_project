from brian2 import *
import numpy as np
import random

defaultclock.dt = 100*ms
N = 10
taum = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0
vt = -45
vr = -60
El = -74
taue = 5*ms
F = 40*Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt = (-(v+60)+ge*(Ee-vr)) / taum : 1
dge/dt = -ge / taue : 1

'''

input = PoissonGroup(60, rates=F,clock = Clock(defaultclock.dt))
input_A=input[0:30] # separting into two groups for the two binary inputs (0,1)
input_B=input[31:60]
hidden = NeuronGroup(4, eqs_neurons,clock = Clock(defaultclock.dt), threshold='v>vt', reset='v = vr',
                      method='linear')
output = NeuronGroup(1, eqs_neurons,clock = Clock(defaultclock.dt), threshold='v>vt', reset='v = vr',
                      method='linear')
hidden.v = -60
output.v = -60

S = Synapses(input, hidden,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )
S.connect()
S.w = 'rand() * gmax'

S2 = Synapses(hidden, output,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )
S2.connect()
S2.w = 'rand() * gmax'

@network_operation()
def monitor(): 
    global o
    if o == -1:
        S.w *= 0.95
        S2.w *= 0.95
    if o == 1:
        S.w *= 1.05
        S2.w *= 1.05

mon = StateMonitor(S, 'w', record=True)
mon2 = StateMonitor(S2, 'w', record=True)

# run epochs
for e in range (10):
    #(0,0) -> 0
    input_A.rates = 0*Hz
    input_B.rates = 0*Hz
    o = -1
    run(500*ms) 
    #(1,0) -> 1
    input_A.rates = 40*Hz
    input_B.rates = 0*Hz
    o = 1
    run(500*ms)
    #(0,1) -> 1
    input_A.rates = 0*Hz
    input_B.rates = 40*Hz
    o = 1
    run(500*ms) 
    #(1,1) -> 0
    input_A.rates = 40*Hz
    input_B.rates = 40*Hz
    o = -1
    run(500*ms)

subplot(211)
plot(mon.t/second, mon.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
subplot(212)
plot(mon2.t/second, mon2.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
show()