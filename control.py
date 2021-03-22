import os
import numpy as np
from Network import Network

##################################################### Data ##################################################################
fromFile = np.loadtxt(fname='training_data_1.csv', delimiter=' ')
X1 = fromFile[:,0:5]

fromFile = np.loadtxt(fname='test_data.csv', delimiter=' ')
X3 = fromFile[:,0:5]

####################################################### Control panel ######################################################
Learning_r = 0.05
Moment = 0.001
Bias = 1
Epoches = 200

Neuron_in_topo = [1,0,1,1]
Neuron_in = 3
Neuron_hidden = 4
Topology = [Neuron_in,Neuron_hidden,3]
Activation_fun_topology = ['ident','radial','ident']

Error_measure_frequency = 1

if not os.path.exists('out'):
    os.makedirs('out')
Path = os.path.join(os.getcwd(), 'out')

Name = '1'
########################################################################################################################

X2 = []
for row in X1:
    k = []
    if Neuron_in_topo[0] == 1:
        k.append(row[0])
    if Neuron_in_topo[1] == 1:
        k.append(row[1])
    if Neuron_in_topo[2] == 1:
        k.append(row[2])
    if Neuron_in_topo[3] == 1:
        k.append(row[3])
    k.append(row[4])
    X2.append(k)
X4 = []
for row in X1:
    k = []
    if Neuron_in_topo[0] == 1:
        k.append(row[0])
    if Neuron_in_topo[1] == 1:
        k.append(row[1])
    if Neuron_in_topo[2] == 1:
        k.append(row[2])
    if Neuron_in_topo[3] == 1:
        k.append(row[3])
    k.append(row[4])
    X4.append(k)
  
net = Network(Topology, Activation_fun_topology, Neuron_in_topo, Moment, Learning_r, Bias, Epoches, Error_measure_frequency)
net.train(X2, Name, Path)
net.test(X4, Name, Path)