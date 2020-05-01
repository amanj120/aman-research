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

chords = {}
chord_idx = 0

for c in c_json:
    name = c["name"]
    frets = c["frets"]
    fingers = c["fingers"]
    idx = chord_idx
    chord_idx += 1
    val = np.asarray([-1,-1,-1,-1,-1,-1,-1,-1], dtype='int8')
    st = frets+fingers
    for i in range(8):
        val[i] = int(st[i])
    chords[name] = (idx,val)

chords2 = np.empty((len(chords),8), dtype="int8")
for i in chords.items():
    chords2[i[1][0]] = i[1][1]

songs = List()

for s in s_json:
    cs = s['chords']
    chords_to_nums = np.empty(len(cs), dtype="int8")
    for i,c in enumerate(cs):
        chords_to_nums[i] = chords[c][0]
    songs.append(chords_to_nums) 

@njit
def ctc(cidx0, cidx1, chords):
    c0 = chords[cidx0]
    c1 = chords[cidx1]
    start = [[-1,-1],[-1,-1],[-1,-1],[-1,-1]] #None, None, None, None]
    end = [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
    c0 = (c0[:4], c0[4:])
    c1 = (c1[:4], c1[4:])

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
def chord_lev(x,y, chords_only, chords, diffs):
    chord1 = chords_only[x]
    chord2 = chords_only[y]

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
            if t1[0] == t2[0] and t1[1] == t2[1]: #if the chords are equal, then the cost 
                val = dp[i-1][j-1]
            else:
                cost1 = ctc(t1[0], t1[1], chords)
                cost2 = ctc(t2[0], t2[1], chords)
                replace = dp[i-1][j-1] - cost1 + cost2
                delete = dp[i-1][j] - cost1
                insert = dp[i][j-1] + cost2
                val = min(replace, delete, insert)
            dp[i][j] = val

    distance = dp[l1-1][l2-1]
    diffs[x][y] = distance

@njit(parallel = True)
def gen_song_graph(chords_only, chords, diffs, l):
	for i in prange(l):
	    for j in range(i,l):
	        chord_lev(i,j,chords_only,chords2,diffs)
	        print(i,j)

l = 40#len(songs)
diffs = np.empty((l,l))

print(type(songs))

gen_song_graph(songs, chords2, diffs, l)

print(diffs)

# gen_song_graph.parallel_diagnostics(level=4)