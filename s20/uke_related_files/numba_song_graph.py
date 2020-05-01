import json
import numpy as np
from numba import njit, prange
from numba import types
from numba.typed import Dict
from numba.typed import List
import pandas as pd

fs = open('songs_1.json','r')
s_json = json.load(fs)
fs.close()
print('there are {} songs'.format(len(s_json)))

fc = open('chords.json','r')
c_json = json.load(fc)
fc.close()
print('there are {} chords'.format(len(c_json)))

chords = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.int8[8],
)
songs = []

for c in c_json:
    name = c["name"]
    frets = c["frets"]
    fingers = c["fingers"]
    val = np.asarray([-1,-1,-1,-1,-1,-1,-1,-1], dtype='int8')
    st = frets+fingers
    for i in range(8):
    	val[i] = int(st[i])
    chords[name] = val

for s in s_json:
    songs.append(s) 

chords_only = List()
for item in songs:
	crd = item['chords']
	tcrd = List()
	for c in crd:
		tcrd.append(c)
	chords_only.append(tcrd)

@njit
def ctc(cidx0, cidx1, chords):
    c0 = chords[cidx0]
    c1 = chords[cidx1]
    '''
    So we want to look at each finger, see which fret it started and ended on, and take the manhattan distance
    of that
    Add up all of those
    
    fing[i] represents which finger should be on string i
    fret[i] represents which fret should be pressed on string i
    
    we look at the nonzero fret values, and see the associated finger and string values
    and from this we can construct a map from fingers to string and fret
    
    but how to deal with barring(I think we should just take the first fret that we find a finger for)
    '''
    start = [[-1,-1],[-1,-1],[-1,-1],[-1,-1]] 
    end = [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
    
    c0 = (c0[:4], c0[4:])
    c1 = (c1[:4], c1[4:])

    # print(c0)
    # print(c1)

    for string in range(4):
        start_fret = c0[0][string]
        end_fret = c1[0][string]
        start_finger = c0[1][string]
        end_finger = c1[1][string]
        if(start_finger != 0 and start[start_finger-1][0] == -1 and start[start_finger-1][1] == -1):
            start[start_finger-1][0] = string
            start[start_finger-1][1] = start_fret
        if(end_finger != 0 and end[end_finger-1][0] == -1 and end[end_finger-1][0] == -1):
            end[end_finger-1][0] = string
            end[end_finger-1][1] = end_fret
    # if verbose:
    # print(start)
    # print(end)
    
    cost = 0
    for i in range(4):
        if(end[i][0] == -1):
            cost += 0 #removing a finger (or potentially not using a finger)
        elif(start[i][0] == -1):
            cost += end[i][1] #the cost is just the fret that you place the finger on
        else:
            cost += abs(start[i][0] - end[i][0]) + abs(start[i][1] - end[i][1])
            #manhattan distance between start and end frets/strings
    
    return cost

@njit
def chord_lev(chord1, chord2, chords):
    # there's one less chord transition than there are chords
    l1 = len(chord1) 
    l2 = len(chord2)
    dp = np.zeros((l1,l2))
    for i in range(l1-1):
        dp[i+1][0] = dp[i][0] + ctc(chord1[i], chord1[i+1], chords)
    for j in range(l2-1):
        dp[0][j+1] = dp[0][j] + ctc(chord2[j], chord2[j+1], chords)
    
    for i in range(1, l1):
        for j in range(1,l2):
            t1 = [chord1[i-1], chord1[i]] #transtion a
            t2 = [chord2[j-1], chord2[j]] #transition b
            val = 0
            if t1 == t2: #if the chords are equal, then the cost 
                val = dp[i-1][j-1]
            else:
                    cost1 = ctc(t1[0], t1[1], chords)
                    cost2 = ctc(t2[0], t2[1], chords)
                    replace = dp[i-1][j-1] - cost1 + cost2
                    delete = dp[i-1][j] - cost1
                    insert = dp[i][j-1] + cost2
                    val = min(replace, delete, insert)
            dp[i][j] = val
    return(dp[l1-1][l2-1])


@njit(nopython=True, parallel=True)
def gen_song_graph(songs, chords):
	l = 40#len(songs)
	diffs = np.zeros((l,l))
	for i in prange(l):
	    for j in range(i,l):
	        diffs[i][j] = chord_lev(songs[i], songs[j], chords)
	        print(i,j)
	return diffs

# print(chords_only[0], chords_only[1])
# print(chords['G'])
d = gen_song_graph(chords_only, chords)

# f = open("diffs.txt", 'w')
# f.write(str(d))
# f.close()

pd.DataFrame(d).to_csv("diffs.csv")
