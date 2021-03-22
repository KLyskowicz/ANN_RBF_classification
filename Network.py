import random
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from Neuron import Neuron

class Network(object):

    def __init__(self, layers_topology, activacion_func_topology, neuron_in_topo, momentum=0.2, learning_rate=0.1, bias=1, epoches=1000, neighbour_amount=2, error_measure_frequency=10):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.bias = bias
        self.epoches = epoches
        self.neuron_in_topo = neuron_in_topo
        self.error_measure_frequency = error_measure_frequency
        self.layers_topology = layers_topology
        self.activacion_func_topology = activacion_func_topology
        self.error = 0
        self.error_sum = 0
        self.error_max = 0
        self.error_X = []
        self.error_Y = []
        self.correct_Y = []     #X jest taki sam jak error_X
        self.predicted = []
        self.expected = []
        self.layers = []
        self.last_error_for_epo = 0
        self.last_error = 0
        self.neighbour_amount = neighbour_amount
        self.last_radials = []

############################################### do radialnych #####################################################################
############### k-srednich #########################################
    def new_layer(self):
        self.layers.clear()
        self.layers = [[Neuron(self.layers_topology[i-1], neuron_number+1, self.activacion_func_topology[i], self.neuron_in_topo, self.momentum, self.learning_rate, self.bias) for neuron_number in range(self.layers_topology[i])] for i in range(len(self.layers_topology))]

    def save_centers_weight(self):
        self.last_radials.clear()
        for radial in self.layers[-1]:
            self.last_radials.append(radial.weights)

    def set_best_centers_weight(self):
        for num, xy in enumerate(self.last_radials):
            self.layers[-1][num].weights = xy

    def allocate_data(self, data):
        for one_data in data:
            distance = []
            for radial in self.layers[1]:
                distance.append( radial.get_distance(one_data[0:(len(one_data)-1)]) )
            self.layers[1][ np.argmin(distance, axis=0) ].data.append(one_data[0:(len(one_data)-1)])

    def clear_center_data(self):
        for radial in self.layers[1]:
            radial.data.clear()

    def update_centers(self):
        for radial in self.layers[1]:
            radial.update_center()

    def get_centers_error(self, data_len):
        error = 0
        for center in self.layers[1]:
            error += center.get_error()
        return error/data_len
############ paramert skalujący ####################################
    def parameter_set(self):
        for radial in self.layers[1]:
            close_nei = []
            x = radial.weights[1:]
            for nei in self.layers[1]:
                close_nei.append([nei.get_distance(x), nei.number])
            close_nei.sort()
            new_param = 0
            for nei in close_nei[1:self.neighbour_amount+1]:
                new_param += math.pow( nei[0] , 2 )
            radial.param = math.sqrt(new_param/self.neighbour_amount)

############################################## stare ##################################################################################
    def predict(self, input):   #input jest takiego samego rozmiaru jak pierwsza warstwa
        for neuron, x in zip(self.layers[0], input):
            neuron.output_value = x
        for layer, previous_layer in zip(self.layers[1:], self.layers[0:]):
            for neuron in layer:
                neuron.predict(previous_layer)

    def mean_squared_error(self, expected_output):
        for neuron, output in zip(self.layers[-1], expected_output):
            delta = output - neuron.output_value
            self.error += (delta*delta)/2

    def back_propagation(self, expected_output):
        self.mean_squared_error(expected_output)
        for neuron, output in zip(self.layers[-1], expected_output):
            neuron.output_layer_factor(output)
        # for layer, next_layer in zip(reversed(self.layers[1:-1]), reversed(self.layers[2:])):
        #     for neuron in layer:
        #         neuron.hidden_layer_factor(next_layer)
        # for layer, previous_layer in zip(reversed(self.layers[1:]), reversed(self.layers[0:-1])):
        for neuron in self.layers[-1]:
            neuron.update_weights(self.layers[1])

#################### wykresy ########################################
    def print_error_plot(self, save_0_print_1_none_2, path, name):
        plt.plot(self.error_X, self.error_Y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Wykres obrazujący zmiany wartości funkcji błędu')
        if save_0_print_1_none_2==0:
            name = str(name) + '.png'
            plt.savefig(os.path.join(path, name))
            plt.close()
        elif save_0_print_1_none_2==1:
            plt.show()
            plt.close()

    def print_correct_plot(self, save_0_print_1_none_2, path, name):
        plt.plot(self.error_X, self.correct_Y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Wykres obrazujący zmiany procentowej porawności klasyfikacji obiektów')
        if save_0_print_1_none_2==0:
            name = str(name) + '.png'
            plt.savefig(os.path.join(path, name))
            plt.close()
        elif save_0_print_1_none_2==1:
            plt.show()
            plt.close()

#################### tabelka #######################################
    def check_correct(self, expected_output):
        y = 0
        j = 0
        for neuron, i in zip(self.layers[-1], range(len(self.layers[-1]))):
            if neuron.output_value > y:
                y = neuron.output_value
                j = i
        self.predicted.append(j)
        for i, num in zip(expected_output, range(3)):
            if i == 1:
                self.expected.append(num)

    def sum_correct(self):
        sum = 0
        for y, d in zip(self.predicted, self.expected):
            if y == d:
                sum += 1
        self.correct_Y.append(sum * 100 / len(self.predicted))

    def init_error_resoult(self, path, rand):
        name = os.path.join(path, rand + ".txt")
        test_error_resoult = open(name,"w+")

    def error_resoult_calculate(self):
        self.error_sum += self.error
        if self.error_max < self.error:
            self.error_max = self.error

    def error_resoult_write(self, data_amount, path, rand):
        name = os.path.join(path, rand + ".txt")
        test_error_resoult = open(name,"w+")
        # self.error_sum = self.error_sum/len(self.layers[-1])
        test_error_resoult.write( " Max error: " + str('%f' % (round(self.error_max/len(self.layers[-1]),6))) + '\n')
        self.error_sum = self.error_sum/data_amount
        test_error_resoult.write( " Avarage error: " + str(round(self.error_sum,6)) + '\n')
        test_error_resoult.write( " Correctness: " + str(round(self.correct_Y[0],6)) )
        test_error_resoult.close()

    def error_resoult_write_2(self, path, rand):
        name = os.path.join(path, rand + ".txt")
        test_error_resoult = open(name,"w+")
        test_error_resoult.write( str(self.learning_rate) + '\n')
        test_error_resoult.write( str(self.momentum) + '\n')
        test_error_resoult.write( str(round(self.correct_Y[0],2)) )
        test_error_resoult.close()

    def end_error_resoult(self):
        test_error_resoult.close()

#################################################### trening ######################################################################
    def train(self, learning_data, name, path):      
        ####################################### k-srednich #######################################
        self.new_layer()
        for j in range(100):
            self.new_layer()
            for i in range(20):
                self.clear_center_data()
                self.allocate_data(learning_data)
                self.update_centers() 
                new_error = self.get_centers_error(len(learning_data))
                if (self.last_error_for_epo - 0.00001 < new_error) and (self.last_error_for_epo != 0):
                    break
                self.last_error_for_epo = new_error
            self.last_error_for_epo = 0
            new_error = self.get_centers_error(len(learning_data))
            if (self.last_error > new_error) or (self.last_error == 0):
                last_error2 = self.last_error
                self.last_error = new_error
                self.save_centers_weight()
        self.set_best_centers_weight()
        ###################################### parametr skalujący ################################
        self.parameter_set()
        ##################################### warstwa wyjściowa ##################################
        for i in range(self.epoches):
            self.error = 0
            np.random.shuffle(learning_data)
            X = []
            Y = []
            self.predicted.clear()
            self.expected.clear()
            for row in learning_data:
                X.append(row[0:len(self.layers[0])])
                if row[-1] == 1:
                    Y.append([1,0,0])
                elif row[-1] == 2:
                    Y.append([0,1,0])
                elif row[-1] == 3:
                    Y.append([0,0,1])
            for data_in, data_out in zip(X, Y):
                self.predict(data_in)
                self.back_propagation(data_out)
                self.check_correct(data_out)
            self.error_Y.append(self.error/len(self.layers[-1]))
            self.error_X.append(i)
            self.sum_correct()
        name1 = name+'el'
        name2 = name+'cl'
        self.print_error_plot(0, path, name1)
        self.print_correct_plot(0, path, name2)

    def test(self, learning_data, name, path):
        X = []
        Y = []
        for row in learning_data:
            X.append(row[0:len(self.layers[0])])
            if row[-1] == 1:
                Y.append([1,0,0])
            elif row[-1] == 2:
                Y.append([0,1,0])
            elif row[-1] == 3:
                Y.append([0,0,1])
        self.error_sum = 0
        self.error_max = 0
        self.error_X.clear()
        self.error_Y.clear()
        self.predicted.clear()
        self.expected.clear()
        self.correct_Y.clear()
        for data_in, data_out, i in zip(X, Y, range(len(X))):
            self.error = 0
            self.predict(data_in)
            self.check_correct(data_out)
            self.mean_squared_error(data_out)
            self.error_resoult_calculate()
            self.error_Y.append(self.error/len(self.layers[-1]))
            self.error_X.append(i+1)
        self.sum_correct()
        # self.error_resoult_write(len(learning_data) ,path, name)
        name1 = name + 'et'
        self.print_error_plot(0, path, name1)
        self.resoult(path, name)

#################################################### do latexu ##############################################
    def resoult(self, path, name):
        tabela = open(os.path.join(path, str(name) + "resoult_in_table.txt"),"w+")

        tab = []
        tab.append([0,0,0])
        tab.append([0,0,0])
        tab.append([0,0,0])

        for i, j in zip(self.expected, self.predicted):
            tab[i][j] += 1

        tabela.write('   | A | B | C \n')
        tabela.write(' A | ' + str(tab[0][0]) + ' | ' + str(tab[0][1])  + ' | ' + str(tab[0][2])  + ' \n')
        tabela.write(' B | ' + str(tab[1][0]) + ' | ' + str(tab[1][1])  + ' | ' + str(tab[1][2])  + ' \n')
        tabela.write(' C | ' + str(tab[2][0]) + ' | ' + str(tab[2][1])  + ' | ' + str(tab[2][2])  + ' \n')

        tabela.close()
