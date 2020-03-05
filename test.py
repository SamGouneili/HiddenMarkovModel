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

def viterbi(self, sequence):
    ###########################################
    # Start your code

    M = np.zeros(( len(sequence), self.num_states))
    Prev = np.zeros(( len(sequence), self.num_states), dtype = np.int64)
    #print(M)

    #probability of going to state 0 and emitting the first character in the sequence
    M[0][0] = math.log(self.prior[0]) + math.log(self.emission[0][sequence[0]])
    #probability of going to state 1 and emitting the first character in the sequence
    M[0][1] = math.log(self.prior[1]) + math.log(self.emission[1][sequence[1]])

    for seq_index in range(1, len(sequence)):

        #the current acid in the sequence, A C T or G
        char = sequence[seq_index]

        #probability of emiting this character in state 0
        Emission_0 = math.log(self.emission[0][char])

        #probability of emiting this character in state 1
        Emission_1 = math.log(self.emission[1][char])

        #probability of starting in state 0 and staying in state 0 and emitting this character
        prob_0_to_0 = Emission_0 + math.log(self.transition[0][0]) + M[seq_index - 1][0]

        #probability of starting in state 0 and transitioning to state 1 and emitting this char
        prob_0_to_1 = Emission_1 + math.log(self.transition[0][1]) + M[seq_index - 1][0]

        #probability of starting in state 1 and transitioning to state 0 and emitting this char
        prob_1_to_0 = Emission_0 + math.log(self.transition[1][0]) + M[seq_index - 1][1]

        #probability of starting in state 1 and staying in state 1 and emitting this character
        prob_1_to_1 = Emission_1 + math.log(self.transition[1][1]) + M[seq_index - 1][1]

        #update probability table
        #the more probable event of staying in state 0 vs transitioning from state 1
        if(prob_0_to_0 > prob_1_to_0):
            M[seq_index][0] = prob_0_to_0 # + M[seq_index - 1][0]
        else:
            M[seq_index][0] = prob_1_to_0 # + M[seq_index - 1][1]

        #the more probable event of staying in state 1 vs transitioning from state 0
        if(prob_0_to_1 > prob_1_to_1):
            M[seq_index][1] = prob_0_to_1 # + M[seq_index - 1][0]
        else:
            M[seq_index][1] = prob_1_to_1 # + M[seq_index - 1][1]

        #keep track of which previous state was more likely
        if (prob_0_to_0 > prob_1_to_0):
            Prev[seq_index][0] = 0
        else:
            Prev[seq_index][0] = 1

        if (prob_0_to_1 > prob_1_to_1):
            Prev[seq_index][1] = 0
        else:
            Prev[seq_index][1] = 1

    #print(M)
    #print(Prev)

    #M = probtable, prev = Prev

    path = []

    if (M[0][-1] > M[1][-1]):
        col = 0
    else:
        col = 1

    path.append(col)

    #iterating backwards, using what state is in the current index of Prev as the next destination
    for iter in range (len(M)- 1, 0, -1):

        path.append(Prev[iter][col])

        if Prev[iter][col] == 1:
            col = 1
        else:
            col = 0

    #iterated backwards so we need to flip the list of states
    path.reverse()

    #print(path)

    return path



        states = [0, 1]

        # Forward Section
        forward = []
        previous_forward = {}
        for i, sequence_i in enumerate(sequence):
            current_forward = {}
            for state in states: #[0, 1]
                if i == 0:
                    previousSumForward = self.prior[state]   # Base Case
                else:
                    previousSumForward = sum(previous_forward[k] * self.transition[k][state] for k in states)
                    # print(previousSumForward)
                    # previousSumForward = self.log_sum(previous_forward[k] + math.log(self.transition[k][state]) for k in states
                current_forward[state] = self.emission[state][sequence_i] * previousSumForward

            forward.append(current_forward)
            previous_forward = current_forward



        p_forward = sum(current_forward[k] * 0.01 for k in states)



        # Backward Section
        backward = []
        previous_backward = {}

        reverse =''.join(reversed(sequence))

        for i, sequence_i_plus in enumerate(reverse):
            current_backward = {}
            for state in states:
                if i == 0:
                    current_backward[state] = 0.01   # Base Case
                else:
                    current_backward[state] = sum(self.transition[state][l] * self.emission[l][sequence_i_plus] * previous_backward[l] for l in states)

            backward.insert(0, current_backward)

            previous_backward = current_backward


        p_backward = sum(self.prior[l] * self.emission[l][sequence[0]] * current_backward[l] for l in states)

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
            posterior.append(forward[i][state] * backward[i][state] / p_forward for state in states)

        return posterior
