import numpy as np

import collections



def performregions(mask, roids):
    x8 = [ 0, -1, 0, 1, -1, -1, 1,  1]
    y8 = [-1,  0, 1, 0, -1,  1, 1, -1]

    h, w = mask.shape
    print("ma_arr", mask.shape)
    print("A")
    vis = [False for i in range(mask.shape[0]*mask.shape[1])]
    mask_out = np.zeros(mask.shape, dtype=int)
    newroids = []
    for r in range(len(roids)):
        node = roids[r]
        for id in range(len(node[0])):
            x = node[0][id]
            y = node[1][id]
            #i = y*w+x;
            i = x*w+y
            #mask_out[x][y] = (r+1)
            l = mask[x][y]
            #print("(%s,%s)"%(x,y))
            if vis[i] == False:
                #print("l",l, i)
                vis[i] = True;
                deq = collections.deque()
                vecx, vecy = [],[]

                deq.appendleft((x,y))
                vecx.append(x)
                vecy.append(y)
                while deq:
                    ix, iy = deq.popleft()
                    for xd, yd in zip(x8, y8):
                        xc = ix+xd
                        yc = iy+yd
                        #j = yc*w+xc
                        j = xc*w+yc
                        if xc>=0 and xc<h and yc>=0 and yc<w and mask[xc][yc]==l and vis[j]==False:
                            vis[j] = True;
                            deq.appendleft((xc,yc))
                            vecx.append(xc)
                            vecy.append(yc)
                #print(vecx,vecy)
                # add regions with at least 100 pixels
                if len(vecx)>100:
                    newroids.append((np.array(vecx), np.array(vecy)))
    del vis

    #print("newroids", newroids, len(newroids))
    #print("newroids", newroids)

    for r in range(len(newroids)):
        node = newroids[r]
        mask_out[node] = (r+1)
    print("B")

    return mask_out, newroids