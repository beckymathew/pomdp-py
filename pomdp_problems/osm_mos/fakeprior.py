import pickle
import numpy as np
import json

pomdp_to_map_fp = "/home/becky/repo/spatial-lang/nyc_3obj/4_1_pomdp_cell_to_map_idx.json"
idx_to_cell_fp = "/home/becky/repo/spatial-lang/nyc_3obj/idx_to_cell_nyc_3obj.json"

def create(m, ranges, val, pomdp_to_map):
    # ranges are pomdp indices

    rang1, rang2 = ranges
    a1,b1 = rang1
    a2,b2 = rang2
    
    m[a1:b1, a2:b2] = val
    fakeprior = {}
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            pomdp_tup_key = str((x,y))
            if pomdp_tup_key in pomdp_to_map:
                map_index = pomdp_to_map[pomdp_tup_key]
                fakeprior[map_index] = m[x,y]
    return fakeprior


if __name__ == "__main__":
    m = np.full((400,400), 0.0)

    pomdp_to_map = {}
    with open(pomdp_to_map_fp, 'r') as fin:
        pomdp_to_map = json.load(fin)

    prior0 = create(m, ((10,20), (20, 50)), 1.0, pomdp_to_map)
    prior1 = create(m, ((0,40), (0, 40)), 1.0, pomdp_to_map)
    prior2 = create(m, ((30,50), (0, 100)), 1.0, pomdp_to_map)

    # write pickle
    with open("random_idx.pkl", "wb") as f:
        pickle.dump({0:prior0,
                     1:prior1,
                     2:prior2}, f)
    
