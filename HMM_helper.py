########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 HMM helper
########################################

import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud
#from matplotlib import animation
#from matplotlib.animation import FuncAnimation


####################
# WORDCLOUD FUNCTIONS
####################

def mask():
    # Parameters.
    r = 128
    d = 2 * r + 1

    # Get points in a circle.
    y, x = np.ogrid[-r:d-r, -r:d-r]
    circle = (x**2 + y**2 <= r**2)

    # Create mask.
    mask = 255 * np.ones((d, d), dtype=np.uint8)
    mask[circle] = 0

    return mask

def text_to_wordcloud(text, max_words=50, title='', show=True):
    plt.close('all')

    # Generate a wordcloud image.
    wordcloud = WordCloud(random_state=0,
                          max_words=max_words,
                          background_color='white',
                          mask=mask()).generate(text)

    # Show the image.
    if show:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=24)
        plt.show()

    return wordcloud

def states_to_wordclouds(hmm, obs_map, max_words=50, show=True):
    # Initialize.
    M = 100000
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = []

    # Generate a large emission.
    emission, states = hmm.generate_emission_t(M, obs_map_r)

    # For each state, get a list of observations that have been emitted
    # from that state.
    obs_count = []
    for i in range(n_states):
        obs_lst = np.array(emission)[np.where(np.array(states) == i)[0]]
        obs_count.append(obs_lst)

    # For each state, convert it into a wordcloud.
    for i in range(n_states):
        obs_lst = obs_count[i]
        sentence = [obs_map_r[j] for j in obs_lst]
        sentence_str = ' '.join(sentence)

        wordclouds.append(text_to_wordcloud(sentence_str, max_words=max_words, title='State %d' % i, show=show))

    return wordclouds


####################
# HMM FUNCTIONS
####################

def parse_observations(text):
    '''
    This function tokenizes the words. It separates punctuation from words (expect apostrophes and hyphens).
    It doesn't allow invalid_symbols to appear in the middle of a sentence; other symbols that aren't valid
    show up as separate tokens. Returns a map from observations to indices and a list of indices corresponding
    to the sequences. 
    Also basically deletes every number in the texts.
     not anymore ALSO IGNORES PUNCTUATION AT THE END OF THE LINE THIS IS IMPORTANT
    '''
    # Convert text to dataset.
    # Symbols that won't be split from words
    valid_symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',\
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '-']
    # Symbols corresponding to punctuation at the end of a line
    end_punctuation = [',', '.', ':', '?', ')']
    # Ignore all the numbers
    invalid_symbols = []
    for i in range(200):
        invalid_symbols.append(str(i))
    lines = [line.split() for line in text.split('\n') if line.split()]
    obs_counter = 0
    obs = []
    obs_map = {}
    
    for line in lines:
        obs_elem = []
        
        for word in line:
            #word = re.sub(r'[^\w]', '', word).lower()
            # Make the word lowercase, keep track of punctuation character
            word = word.lower()
            check1 = False
            bad_char = ''
            if word not in obs_map:
                check1 = False
                # Check that the word is completely valid
                for char in word:
                    if char not in valid_symbols:
                        check1 = True
                        bad_char = char
                        break
                if check1:
                    # Remove the punctuation
                    word = word.replace(bad_char, '')
                    if word not in obs_map and word not in invalid_symbols:
                        # Add unique words to the observations map.
                        obs_map[word] = obs_counter
                        obs_counter += 1
                    # Add the punctuation as a symbol if it's not invalid
                    if (bad_char not in obs_map) and (bad_char not in invalid_symbols):
                        obs_map[bad_char] = obs_counter
                        obs_counter += 1
                else:
                    # Add unique words to the observations map.
                    obs_map[word] = obs_counter
                    obs_counter += 1
                        # Add the encoded word.
            # Append the word if not invalid
            if word not in invalid_symbols:
                obs_elem.append(obs_map[word])
            # If punctuation, add the punctuation as a symbol
            if bad_char != '' and bad_char not in invalid_symbols:
                obs_elem.append(obs_map[bad_char])
        # Add spaces at the end of lines without any punctuation to indicate no end punctuation.
        if obs_elem != [] and obs_map_reverser(obs_map)[obs_elem[-1]] not in end_punctuation:
            if ' ' not in obs_map:
                obs_map[' '] = obs_counter
                obs_counter += 1
            obs_elem.append(obs_map[' '])
        # Add the encoded sequence.
        obs.append(obs_elem)

    return obs, obs_map

def parse_ends(text):
    '''
    Converts the end punctuations into sets of sequences
    '''
    # Convert text to dataset.
    lines = [line.split() for line in text.split('\n') if line.split()]
    endlist = []
    for line in lines:
        endlist.append(line[-1][-1])
    valid_ends = [',', '.', '?', '!', ':', ';']
    valid_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',\
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    end_map = {}
    sequences = []
    sequence = []
    map_ind = 0
    i = 0
    num_valids = 0
    #print(endlist)
    print(len(endlist))
    while 1:
        if endlist[i] in valid_ends:
            if endlist[i] not in end_map:
                end_map[endlist[i]] = map_ind
                map_ind += 1
            num_valids += 1
            sequence.append(endlist[i])
        if endlist[i] in valid_letters:
            if ' ' not in end_map:
                end_map[' '] = map_ind
                map_ind += 1
            num_valids += 1
            sequence.append(' ')
        i += 1
        if num_valids == 14:
            sequences.append(sequence)
            sequence = []
            num_valids = 0
        if i == len(endlist):
            break
    return sequences, end_map
    

def obs_map_reverser(obs_map):
    obs_map_r = {}

    for key in obs_map:
        obs_map_r[obs_map[key]] = key

    return obs_map_r

def sample_sentence(hmm, obs_map, n_words=100):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.generate_emission(n_words, obs_map_r)
    sentence = ''
    valid_symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',\
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '-', 
                    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    for i in range(len(emission)):
        space = True
        for char in obs_map_r[emission[i]]:
            # Don't add a space if it's punctuation.
            if char not in valid_symbols:
                space = False
                break
        if space:
            sentence += ' '
        sentence += obs_map_r[emission[i]]
    # Ignore the first space and capitalize.
    return sentence[1:].capitalize()


####################
# HMM VISUALIZATION FUNCTIONS
####################

def visualize_sparsities(hmm, O_max_cols=50, O_vmax=0.1):
    plt.close('all')
    plt.set_cmap('viridis')

    # Visualize sparsity of A.
    plt.imshow(hmm.A, vmax=1.0)
    plt.colorbar()
    plt.title('Sparsity of A matrix')
    plt.show()

    # Visualize parsity of O.
    plt.imshow(np.array(hmm.O)[:, :O_max_cols], vmax=O_vmax, aspect='auto')
    plt.colorbar()
    plt.title('Sparsity of O matrix')
    plt.show()


####################
# HMM ANIMATION FUNCTIONS
####################

def animate_emission(hmm, obs_map, M=8, height=12, width=12, delay=1):
    # Parameters.
    lim = 1200
    text_x_offset = 40
    text_y_offset = 80
    x_offset = 580
    y_offset = 520
    R = 420
    r = 100
    arrow_size = 20
    arrow_p1 = 0.03
    arrow_p2 = 0.02
    arrow_p3 = 0.06
    
    # Initialize.
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = states_to_wordclouds(hmm, obs_map, max_words=20, show=False)

    # Initialize plot.    
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.grid('off')
    plt.axis('off')
    ax.set_xlim([0, lim])
    ax.set_ylim([0, lim])

    # Plot each wordcloud.
    for i, wordcloud in enumerate(wordclouds):
        x = x_offset + int(R * np.cos(np.pi * 2 * i / n_states))
        y = y_offset + int(R * np.sin(np.pi * 2 * i / n_states))
        ax.imshow(wordcloud.to_array(), extent=(x - r, x + r, y - r, y + r), aspect='auto', zorder=-1)

    # Initialize text.
    text = ax.text(text_x_offset, lim - text_y_offset, '', fontsize=24)
        
    # Make the arrows.
    zorder_mult = n_states ** 2 * 100
    arrows = []
    for i in range(n_states):
        row = []
        for j in range(n_states):
            # Arrow coordinates.
            x_i = x_offset + R * np.cos(np.pi * 2 * i / n_states)
            y_i = y_offset + R * np.sin(np.pi * 2 * i / n_states)
            x_j = x_offset + R * np.cos(np.pi * 2 * j / n_states)
            y_j = y_offset + R * np.sin(np.pi * 2 * j / n_states)
            
            dx = x_j - x_i
            dy = y_j - y_i
            d = np.sqrt(dx**2 + dy**2)

            if i != j:
                arrow = ax.arrow(x_i + (r/d + arrow_p1) * dx + arrow_p2 * dy,
                                 y_i + (r/d + arrow_p1) * dy + arrow_p2 * dx,
                                 (1 - 2 * r/d - arrow_p3) * dx,
                                 (1 - 2 * r/d - arrow_p3) * dy,
                                 color=(1 - hmm.A[i][j], ) * 3,
                                 head_width=arrow_size, head_length=arrow_size,
                                 zorder=int(hmm.A[i][j] * zorder_mult))
            else:
                arrow = ax.arrow(x_i, y_i, 0, 0,
                                 color=(1 - hmm.A[i][j], ) * 3,
                                 head_width=arrow_size, head_length=arrow_size,
                                 zorder=int(hmm.A[i][j] * zorder_mult))

            row.append(arrow)
        arrows.append(row)

    emission, states = hmm.generate_emission(M)

    def animate(i):
        if i >= delay:
            i -= delay

            if i == 0:
                arrows[states[0]][states[0]].set_color('red')
            elif i == 1:
                arrows[states[0]][states[0]].set_color((1 - hmm.A[states[0]][states[0]], ) * 3)
                arrows[states[i - 1]][states[i]].set_color('red')
            else:
                arrows[states[i - 2]][states[i - 1]].set_color((1 - hmm.A[states[i - 2]][states[i - 1]], ) * 3)
                arrows[states[i - 1]][states[i]].set_color('red')

            # Set text.
            text.set_text(' '.join([obs_map_r[e] for e in emission][:i+1]).capitalize())

            return arrows + [text]

    # Animate!
    print('\nAnimating...')
    anim = FuncAnimation(fig, animate, frames=M+delay, interval=1000)

    return anim

    # honestly this function is so jank but who even fuckin cares
    # i don't even remember how or why i wrote this mess
    # no one's gonna read this
    # hey if you see this tho hmu on fb let's be friends
