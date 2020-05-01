import json
import numpy as np
from numba import njit, prange, cuda
from numba import types
from numba.typed import Dict
from numba.typed import List
from numba import int8, int16
import pandas as pd

num_songs = 5835

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
    val = np.asarray([-1,-1,-1,-1,-1,-1,-1,-1], dtype="int8")
    st = frets+fingers
    for i in range(8):
        val[i] = int(st[i])
    chords[name] = (idx,val)

chords2 = np.empty((len(chords),8), dtype="int8")
for i in chords.items():
    chords2[i[1][0]] = i[1][1]

songs = np.empty(shape=(num_songs, 400), dtype="int16")

for i,s in enumerate(s_json):
    cs = s['chords']
    lencs = len(cs)
    songs[i][399] = lencs
    # chords_to_nums = np.empty(len(cs), dtype="int16")
    for j,c in enumerate(cs):
        songs[i][j] = chords[c][0]
    # songs.append(chords_to_nums) 


@cuda.jit('int16(int16,int16,int8[:,:])', device=True)
def ctc(cidx0, cidx1, chords):
    c0 = chords[cidx0]
    c1 = chords[cidx1]
    startarr = cuda.local.array(shape=(4,2), dtype=int8) #[[-1,-1],[-1,-1],[-1,-1],[-1,-1]] #None, None, None, None]
    endarr = cuda.local.array(shape=(4,2), dtype=int8) #[[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
    # c0 = [c00[0:4], c00[4:]]
    # c1 = [c01[0:4], c01[4:]]
    '''
    c0 = [fret fret fret fret finger finger finger finger]
    c1 = [fret fret fret fret finger finger finger finger]

    we need to convert this to finger finger finger finger
                               string string string string
                               fret   fret   fret   fret
    '''
    # finger_1 = (0,0,0,0)
    # finger_2 = (0,0,0,0)
    # finger_3 = (0,0,0,0)
    # finger_4 = (0,0,0,0)
    for i in range(4):
        startarr[i][0] = -1
        endarr[i][0]   = -1
        startarr[i][1] = -1
        endarr[i][1]   = -1

    for ss in range(4):
        start_fret = c0[ss]
        end_fret = c1[ss]
        start_finger = c0[4+ss]
        end_finger = c1[4+ss]
        if(start_finger != 0 and startarr[start_finger-1][0] == -1 and startarr[start_finger-1][1] == -1):
            startarr[start_finger-1][0] = ss
            startarr[start_finger-1][1] = start_fret
        if(end_finger != 0 and endarr[end_finger-1][0] == -1 and endarr[end_finger-1][0] == -1):
            endarr[end_finger-1][0] = ss
            endarr[end_finger-1][1] = end_fret
    
    cost = 0
    for i in range(4):
        if(endarr[i][0] == -1):
            cost += 0 #removing a finger (or potentially not using a finger)
        elif(startarr[i][0] == -1):
            cost += endarr[i][1] #the cost is just the fret that you place the finger on
        else:
            cost += abs(startarr[i][0] - endarr[i][0]) + abs(startarr[i][1] - endarr[i][1])
            #manhattan distance between start and end frets/sss
    return cost
    # return 12

@cuda.autojit
def chord_lev(chords_only, chords, diffs, lenco):
    # x, y = cuda.grid(2)
    # len(chords_only) = 5835
    pos = cuda.threadIdx.x + (cuda.blockIdx.x * 32)

    y = 0
    x = pos
    level = lenco - 1
    while(x >= level and level > 0):
        x -= level
        level -= 1
        y += 1
    # if(x < lenco and y < lenco-1):
    #     y = lenco - y - 1
    #     print(x,y)

    if(x < lenco and y < lenco-1):
        y = lenco - y - 1
        chord1 = chords_only[x]
        chord2 = chords_only[y]

        l1 = chords_only[x][399] #len(chord1) 
        l2 = chords_only[y][399] #len(chord2)
        dp = cuda.local.array(shape=(2,400), dtype=int16) #sliding window DP table
        
        dp[0][0] = 0


        row = 0;#current roe
        nrow = 1; # next row

        for j in range(l2-1):
            dp[row][j+1] = dp[row][j] + ctc(chords_only[y][j], chords_only[y][j+1], chords)
            # initialize top row

        for i in range(1,l1):
            row = 1 - row
            nrow = 1 - nrow
            dp[row][0] = ctc(chords_only[x][i-1],chords_only[x][i], chords) # initialize left hand side
            for j in range(1,l2):
                t11 = chords_only[x][i-1]
                t12 = chords_only[x][i] #transtion a
                t21 = chords_only[y][j-1]
                t22 = chords_only[y][j] #transition b
                val = 0
                if t11 == t21 and t12 == t22: #if the chords are equal, then the cost 
                    val = dp[nrow][j-1]
                else:
                    cost1 = ctc(t11, t12, chords)
                    cost2 = ctc(t21, t22, chords)
                    replace = dp[nrow][j-1] - cost1 + cost2
                    delete = dp[nrow][j] - cost1
                    insert = dp[row][j-1] + cost2
                    val = min(replace, delete, insert)
                dp[row][j] = val


        # for i in range(l1-1):
        #     dp[i+1][0] = dp[i][0] + ctc(chords_only[x][i],chords_only[x][i+1], chords)
        # for j in range(l2-1):
        #     dp[0][j+1] = dp[0][j] + ctc(chords_only[y][j], chords_only[y][j+1], chords)
        # for i in range(1, l1):
        #     for j in range(1,l2):
        #         t11 = chords_only[x][i-1]
        #         t12 = chords_only[x][i] #transtion a
        #         t21 = chords_only[y][j-1]
        #         t22 = chords_only[y][j] #transition b
        #         val = 0
        #         if t11 == t21 and t12 == t22: #if the chords are equal, then the cost 
        #             val = dp[i-1][j-1]
        #         else:
        #             cost1 = ctc(t11, t12, chords)
        #             cost2 = ctc(t21, t22, chords)
        #             replace = dp[i-1][j-1] - cost1 + cost2
        #             delete = dp[i-1][j] - cost1
        #             insert = dp[i][j-1] + cost2
        #             val = min(replace, delete, insert)
        #         dp[i][j] = val
        # distance = dp[l1-1][l2-1]

        diffs[x][y] = dp[row][l2-1]
        # diffs[x][y] = x*1000 + y
        # diffs[x][y] = l1
        # diffs[x][y] = cuda.threadIdx.x
        # diffs[x][y] = pos
        # diffs[x][y] = 1

l = len(songs)
diffs = np.zeros(shape=(l,l),dtype="int32")
print(chords2[songs[0][0]])
print(chords2[songs[0][1]])

d_songs = cuda.to_device(songs)
d_chords2 = cuda.to_device(chords2)
d_diffs = cuda.to_device(diffs)

num_blocks = ((l*l)//64) + 1 # divide by 32, and then divide than number in half

chord_lev[num_blocks,32](d_songs, d_chords2, d_diffs, l)

d_diffs.copy_to_host(diffs)

print(diffs)

pd.DataFrame(diffs).to_csv("diffs.csv")
