########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np
from numba import jit, jitclass
from tqdm import tqdm

def argmax(x):
    return x.index(max(x))


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        Uses:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.

        # These are in observation_index, state_index order
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]
            
        ###
        ###
        ### 

        for state in range(self.L):
            # O[i, j] is the probability of observing j given state i.
            # A[i, j] is the probability of transiting from i to j
            probs[0][state] = self.A_start[state] * self.O[state][x[0]]
            seqs[0][state] = state
            # Probs[i, j] = time i , state j
        for ts in range(1, M):
            for state in range(self.L):
                tmp = [probs[ts - 1][state_2] * self.O[state][x[ts]] * self.A[state_2][state] for state_2 in range(self.L)]
                probs[ts][state] = max(tmp)
                seqs[ts][state] = str(seqs[ts-1][argmax(tmp)]) + str(state)#argmax(tmp)

        ###
        ###
        ###
    
        idx = M - 1
        end_state = argmax(probs[idx])
        return seqs[idx][end_state]

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        Uses:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        M = len(x)      # Length of sequence.
        probs = np.array([[0. for _ in range(self.L)] for _ in range(M + 1)])
            
        ###
        ###
        ### 

        for state in range(self.L):
            # O[i, j] is the probability of observing j given state i.
            # A[i, j] is the probability of transiting from i to j
            probs[0][state] = self.A_start[state] * self.O[state][x[0]]
            # Probs[i, j] = time i , state j
        if(normalize):
            probs[0] = probs[0] / np.sum(probs[0])
        for ts in range(1, M):
            for state in range(self.L):
                tmp = [probs[ts - 1][state_2] * self.O[state][x[ts]] * self.A[state_2][state] for state_2 in range(self.L)]
                probs[ts][state] = sum(tmp)
            if(normalize):
                probs[ts] = probs[ts] / np.sum(probs[ts])

        ###
        ###
        ###
        

        return probs[:-1]


    
    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        Uses:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        M = len(x)      # Length of sequence.
        probs = np.array([[0. for _ in range(self.L)] for _ in range(M + 1)], dtype=np.float64)
            
        ###
        ###
        ### 

        for state in range(self.L):
            probs[-1][state] = 1
        if(normalize):
            probs[-1] = probs[-1] / np.sum(probs[-1])
        for ts in range(M-1, 0, -1):
            for state in range(self.L):
                # probs[ts-1][state] = 0
                # print(M,ts, probs[ts], probs[ts-1])
                tmp = [probs[ts+1][state_2] * self.O[state_2][x[ts]] * self.A[state][state_2] for state_2 in range(self.L)]
                probs[ts][state] = np.sum(tmp)
            if(normalize):
                probs[ts] = probs[ts] / np.sum(probs[ts])
        # print(probs)
        return probs

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        ###
        ###
        ### 
        N = len(X)
        
        
        A = np.zeros((self.L, self.L))
        for a in range(self.L):
            for b in range(self.L):
                numerator = 0.0
                denominator = 0.0
                for j in range(N):
                    M = len(X[j])
                    for i in range(M-1):
                        if(Y[j][i+1] == b and Y[j][i] == a):
                            numerator += 1.0
                        if(Y[j][i] == a):
                            denominator+=1.0
                A[a,b] = numerator/denominator
        O = np.zeros((self.L, self.D))         
        for z in range(self.L):
            for w in range(self.D):
                numerator = 0.0
                denominator = 0.0
                for j in range(N):
                    M = len(X[j])
                    for i in range(M):
                        
                        if(X[j][i] == w and Y[j][i] == z):
                            numerator += 1.0
                        if(Y[j][i] == z):
                            denominator+=1.0
                O[z,w] = numerator/denominator

        self.A = A.tolist()
        self.O = O.tolist()

        pass

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        @jit
        def fast_compute(M, alpha, beta, eps, L, A_num, A_den, A, O, x, D, O_num, O_den):
            for i in range(0, M):
                sum_O = np.sum(alpha[i]*beta[i])
                sum_den = np.sum(alpha[i-1]*beta[i-1])
                if(i >= 1):
                    # outer_prod1 = np.sum(np.outer(alpha[i-combo[0]],beta[i-combo[1]]))
                    sum_1 = eps
                    for ap in range(L):
                        for bp in range(L):
                            sum_1 += alpha[i-1][ap] * beta[i][bp]*A[ap][bp]*O[bp][x[i]]

                    for a in range(L):
                        for b in range(L): 
                            A_num[a][b] += (alpha[i-1][a]*beta[i][b]*A[a][b]*O[b][x[i]]) / sum_1
                            A_den[a][b] += (alpha[i-1][a]*beta[i-1][a]) / (sum_den + eps)
                # Observation

                for z in range(L):
                    for w in range(D):
                        
                        O_num[z][w] += (x[i] == w)*(alpha[i][z]*beta[i][z]) / (sum_O + eps)
                        O_den[z][w] += (alpha[i][z]*beta[i][z]) / (sum_O + eps)
            return A_num, A_den, O_num, O_den


        eps = 0
        N = len(X)
        A = np.array(self.A)
        O = np.array(self.O)  

        # N_iters = 1
        for iter in tqdm(range(N_iters)):
            A_num = np.zeros_like(A, dtype = np.float64)
            A_den = np.zeros_like(A, dtype = np.float64)
            O_num = np.zeros_like(O, dtype = np.float64)
            O_den = np.zeros_like(O, dtype = np.float64)
            alphas = []
            betas = []

            for x in X:
                alpha = np.array(self.forward(x, normalize=True), dtype = np.float64)
                beta = np.array(self.backward(x, normalize=True), dtype = np.float64)
                alphas.append(alpha)
                betas.append(beta[1:])
            # O[i, j] is the probability of observing j given state i.
            # A[i, j] is the probability of transiting from i to j

            for j in (range(N)):
                M = len(X[j])
                x = X[j]
                alpha = alphas[j]
                beta = betas[j]
                A_num, A_den, O_num, O_den = fast_compute(M, alpha, beta, eps, self.L, A_num, A_den, A, O, x, self.D, O_num, O_den)

            A = (A_num / (A_den + eps))
            O = O_num / (O_den + eps)
            self.A = A.tolist()
            self.O = O.tolist()

        pass

    def generate_emission(self, M, obs_map):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        valid_symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',\
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '-']
        

        ###
        ###
        ### 
        # O[i, j] is the probability of observing j given state i.
        # A[i, j] is the probability of transiting from i to j
        i = 0
        while i != M:
            if(len(states) == 0):
                start_state = np.random.randint(self.L)
                probs = self.O[start_state] / np.sum(self.O[start_state])
                obs = np.random.choice(np.arange(self.D), p = probs)
                states.append(start_state)
                emission.append(obs)
                inc = True
                # If punctuation, generate extra symbols because they aren't words.
                for char in obs_map[obs]:
                    if char not in valid_symbols:
                        inc = False
                        break
                if inc:
                    i += 1
                    
            else:
                probs = self.A[states[-1]] / np.sum(self.A[states[-1]])
                state = np.random.choice(np.arange(self.L), p = probs)
                probs = self.O[state] / np.sum(self.O[state])
                obs = np.random.choice(np.arange(self.D), p = probs)
                states.append(state)
                emission.append(obs)
                inc = True
                for char in obs_map[obs]:
                    if char not in valid_symbols:
                        inc = False
                        break
                if inc:
                    i += 1
        ###
        ###
        ###

        return emission, states
    
    def generate_emission_t(self, M, obs_map):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        This function is specifically used for wordcloud generation to track the
        progress in emission generation.
        '''

        emission = []
        states = []
        valid_symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',\
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '-']
        

        ###
        ###
        ### 
        # O[i, j] is the probability of observing j given state i.
        # A[i, j] is the probability of transiting from i to j
        i = 0
        while i != M:
            if (i % 10000 == 0):
                print("Completed: {}".format(i / M))
            if(len(states) == 0):
                start_state = np.random.randint(self.L)
                probs = self.O[start_state] / np.sum(self.O[start_state])
                obs = np.random.choice(np.arange(self.D), p = probs)
                states.append(start_state)
                emission.append(obs)
                inc = True
                for char in obs_map[obs]:
                    if char not in valid_symbols:
                        inc = False
                        break
                if inc:
                    i += 1
                    
            else:
                probs = self.A[states[-1]] / np.sum(self.A[states[-1]])
                state = np.random.choice(np.arange(self.L), p = probs)
                probs = self.O[state] / np.sum(self.O[state])
                obs = np.random.choice(np.arange(self.D), p = probs)
                states.append(state)
                emission.append(obs)
                inc = True
                for char in obs_map[obs]:
                    if char not in valid_symbols:
                        inc = False
                        break
                if inc:
                    i += 1
        ###
        ###
        ###

        return emission, states

    def generate_ends(self, init_state, obs_map, obs_map_r, cap=False):
        '''
        Generates an end punctuation given the current state of the HMM.
        '''


        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2F)
        ###
        ###
        ###
        # Always select period.
        if cap:
            punctuations = ['.']
        else:
            # Valid end punctuations
            punctuations = [',', '.', '?', ':', ';', ' ']
        # Get observation indices of punctuation
        puc_idx = []
        for puc in punctuations:
            puc_idx.append(obs_map[puc])
        puc_weights = []
        # Append the probabilities
        for puc_id in puc_idx:
            puc_weights.append(self.O[init_state][puc_id])
        # Normalize probabilities
        puc_weights = [puc_weights[i] / sum(puc_weights) for i in range(len(puc_weights))]
        # Choose end punctuation and return it
        return obs_map_r[np.random.choice(puc_idx, p = puc_weights)]

    def generate_from_list(self, init_state, lst, obs_map, obs_map_r):
        '''
        This function picks a random element from a list given the state it's in.
        '''
        # Get object indices of elements
        ob_idx = []
        for word in lst:
            ob_idx.append(obs_map[word])
        ob_weights = []
        # Append objects' emission probabilities
        for ob_id in ob_idx:
            ob_weights.append(self.O[init_state][ob_id])
        # Normalize probabilities and choose an element.
        ob_weights = [ob_weights[i] / sum(ob_weights) for i in range(len(ob_weights))]
        return obs_map_r[np.random.choice(ob_idx, p = ob_weights)]


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = max(observations) + 1
    print(D)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM