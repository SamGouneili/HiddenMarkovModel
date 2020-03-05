#!/usr/bin/python3

import sys
import random
import math

import numpy as np
import operator

#####################################################
#####################################################
# Please enter the number of hours you spent on this
# assignment here
num_hours_i_spent_on_this_assignment = 20
#####################################################
#####################################################

#####################################################
#####################################################
# Give one short piece of feedback about the course so far. What
# have you found most interesting? Is there a topic that you had trouble
# understanding? Are there any changes that could improve the value of the
# course to you? (We will anonymize these before reading them.)
# <Your feedback goes here>
#####################################################
#####################################################



# Outputs a random integer, according to a multinomial
# distribution specified by probs.
def rand_multinomial(probs):
    # Make sure probs sum to 1
    assert(abs(sum(probs) - 1.0) < 1e-5)
    rand = random.random()
    for index, prob in enumerate(probs):
        if rand < prob:
            return index
        else:
            rand -= prob
    return 0

# Outputs a random key, according to a (key,prob)
# iterator. For a probability dictionary
# d = {"A": 0.9, "C": 0.1}
# call using rand_multinomial_iter(d.items())
def rand_multinomial_iter(iterator):
    rand = random.random()
    for key, prob in iterator:
        if rand < prob:
            return key
        else:
            rand -= prob
    return 0


class HMM():

    def __init__(self):
        #obseravtions => sequence
        self.num_states = 2
        self.prior      = np.array([0.5, 0.5])
        # self.transition = np.array([[0.999, 0.001], [0.01, 0.99]])
        self.transition = np.array([[0.99, 0.01], [0.01, 0.99]])

        self.emission   = np.array([{"A": 0.291, "T": 0.291, "C": 0.209, "G": 0.209},
                                    {"A": 0.169, "T": 0.169, "C": 0.331, "G": 0.331}])

    # Generates a sequence of states and characters from
    # the HMM model.
    # - length: Length of output sequence
    def sample(self, length):
        sequence = []
        states = []
        rand = random.random()
        cur_state = rand_multinomial(self.prior)
        for i in range(length):
            states.append(cur_state)
            char = rand_multinomial_iter(self.emission[cur_state].items())
            sequence.append(char)
            cur_state = rand_multinomial(self.transition[cur_state])
        return sequence, states

    # Generates a emission sequence given a sequence of states
    def generate_sequence(self, states):
        sequence = []
        for state in states:
            char = rand_multinomial_iter(self.emission[state].items())
            sequence.append(char)
        return sequence




    # Outputs the most likely sequence of states given an emission sequence
    # - sequence: String with characters [A,C,T,G]
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def viterbi(self, sequence):
        ###########################################
        # Start your code

        m = np.zeros((len(sequence), self.num_states))
        prev = np.zeros((len(sequence), self.num_states), dtype = np.int64)

        m[0][0] = math.log(self.prior[0]) + math.log(self.emission[0][sequence[0]])
        m[0][1] = math.log(self.prior[1]) + math.log(self.emission[1][sequence[1]])

        for index in range(1, len(sequence)):

            #the current acid in the sequence, A C T or G
            acid = sequence[index]

            probabilityEmitingState0 = math.log(self.emission[0][acid])
            probabilityEmitingState1 = math.log(self.emission[1][acid])

            #0 -> 0
            probabilityZeroStay = probabilityEmitingState0 + math.log(self.transition[0][0]) + m[index - 1][0]
            #0 -> 1
            probabilityZeroTransition = probabilityEmitingState1 + math.log(self.transition[0][1]) + m[index - 1][0]
            #1 -> 0
            probabilityOneTransition = probabilityEmitingState0 + math.log(self.transition[1][0]) + m[index - 1][1]
            #1 -> 1
            probabilityOneStay = probabilityEmitingState1 + math.log(self.transition[1][1]) + m[index - 1][1]


            #Case 1
            if(probabilityZeroStay > probabilityOneTransition):
                m[index][0] = probabilityZeroStay
            else:
                m[index][0] = probabilityOneTransition

            #Case 2
            if(probabilityZeroTransition > probabilityOneStay):
                m[index][1] = probabilityZeroTransition
            else:
                m[index][1] = probabilityOneStay

            #Case 3
            if (probabilityZeroStay > probabilityOneTransition):
                prev[index][0] = 0
            else:
                prev[index][0] = 1

            #Case 4
            if (probabilityZeroTransition > probabilityOneStay):
                prev[index][1] = 0
            else:
                prev[index][1] = 1


        forwardPath = []

        if (m[0][-1] > m[1][-1]):
            col = 0
        else:
            col = 1
        forwardPath.append(col)

        #Start at end and iterate backwards
        for i in range (len(m)- 1, 0, -1):
            forwardPath.append(prev[i][col])
            if prev[i][col] == 1:
                col = 1
            else:
                col = 0

        #Reverse since we iterated backwards
        forwardPath.reverse()
        return forwardPath

        # End your code
        ###########################################


    def log_sum(self, factors):
        if abs(min(factors)) > abs(max(factors)):
            a = min(factors)
        else:
            a = max(factors)

        total = 0
        for x in factors:
            total += math.exp(x - a)
        return a + math.log(total)

    # - sequence: String with characters [A,C,T,G]
    # return: posterior distribution. shape should be (len(sequence), 2)
    # Please use log_sum() in posterior computations.
def posterior(self, sequence):
    ###########################################
    # Start your code
    states = [0, 1]

    # Forward Section
    forward = []
    previous_forward = {}
    for i, sequence_i in enumerate(sequence):
        current_forward = {}
        for state in states: #[0, 1]
            if i == 0:
                previousSumForward = math.log(self.prior[state])   # Base Case
            else:
                for k in states:
                    if k == 0:
                        arg1 = previous_forward[k] + math.log(self.transition[k][state])
                    else:
                        arg2 = previous_forward[k] + math.log(self.transition[k][state])

                # print(arg1)
                # print(arg2)
                previousSumForward = self.log_sum([arg1, arg2])
                # print(previousSumForward)
                # previousSumForward = self.log_sum(previous_forward[k] + math.log(self.transition[k][state]) for k in states
            current_forward[state] = math.log(self.emission[state][sequence_i]) + previousSumForward

        forward.append(current_forward)
        previous_forward = current_forward


    for k in states:
        if k == 0:
            arg1 = current_forward[k] + math.log(0.01)
        else:
            arg2 = current_forward[k] + math.log(0.01)
    p_forward = self.log_sum([arg1, arg2]) #0 -> end_st



    # Backward Section
    backward = []
    previous_backward = {}

    reverse =''.join(reversed(sequence))

    for i, sequence_i_plus in enumerate(reverse):
        current_backward = {}
        for state in states:
            if i == 0:
                current_backward[state] = math.log(0.01)   # Base Case
            else:
                for l in states:
                    if l == 0:
                        arg1 = math.log(self.transition[state][l]) + math.log(self.emission[l][sequence_i_plus]) + previous_backward[l]
                    else:
                        arg2 = math.log(self.transition[state][l]) + math.log(self.emission[l][sequence_i_plus]) + previous_backward[l]
                current_backward[state] = self.log_sum([arg1, arg2])

        backward.insert(0, current_backward)

        previous_backward = current_backward


    for l in states:
        if l == 0:
            arg1 = math.log(self.prior[l]) + math.log(self.emission[l][sequence[0]]) + current_backward[l]
        else:
            arg2 = math.log(self.prior[l]) + math.log(self.emission[l][sequence[0]]) + current_backward[l]

    p_backward = self.log_sum([arg1, arg2])

    # print(forward)
    # print('\n')
    # print(backward)
    # print('\n')
    # print(p_forward)
    # print('\n')
    # print("Hello")


    # merging the two parts
    posterior = []
    for i in range(len(sequence)):
        posterior.append(forward[i][state] + backward[i][state] - p_forward for state in states)


    return posterior

        # End your code
        ###########################################


    # Output the most likely state for each symbol in an emmision sequence
    # - sequence: posterior probabilities received from posterior()
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def posterior_decode(self, posterior):
        nSamples  = len(sequence)
        post = self.posterior(sequence)
        best_path = np.zeros(nSamples)
        for t in range(nSamples):
            best_path[t], _ = max(enumerate(post[t]), key=operator.itemgetter(1))
        return list(best_path.astype(int))


def read_sequences(filename):
    inputs = []
    with open(filename, "r") as f:
        for line in f:
            inputs.append(line.strip())
    return inputs

def write_sequence(filename, sequence):
    with open(filename, "w") as f:
        f.write("".join(sequence))

def write_output(filename, viterbi, posterior):
    vit_file_name = filename[:-4]+'_viterbi_output.txt'
    with open(vit_file_name, "a") as f:
        for state in range(2):
            f.write(str(viterbi.count(state)))
            f.write("\n")
        f.write(" ".join(map(str, viterbi)))
        f.write("\n")

    pos_file_name = filename[:-4]+'_posteri_output.txt'
    with open(pos_file_name, "a") as f:
        for state in range(2):
            f.write(str(posterior.count(state)))
            f.write("\n")
        f.write(" ".join(map(str, posterior)))
        f.write("\n")


if __name__ == '__main__':

    hmm = HMM()

    file = sys.argv[1]
    sequences  = read_sequences(file)
    for sequence in sequences:
        viterbi   = hmm.viterbi(sequence)
        posterior = hmm.posterior_decode(sequence)
        write_output(file, viterbi, posterior)
