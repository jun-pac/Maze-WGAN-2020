# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import numpy as np
import random
import sys
import copy
import random
import tqdm
import os

sys.setrecursionlimit(10000)


'''
#Code for visualization
a=np.array([[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]])
b=[]
for i in range(4):
  b.append([])
  for j in range(4):
    b[i].append([])
    if a[i][j]==1:
      b[i][j]=[255,255,255]
    else:
      b[i][j]=[0,0,0]
plt.imshow(b)
'''
PATH=os.getcwd()
PATH=PATH+'/'


dir=[[-1,0],[0,1],[1,0],[0,-1]]

def connect_check(p1,p2,n,visited,white_conn=False):
  if(p1==p2):
    return True
  if(visited[p1[0]][p1[1]]):
    return False
  
  visited[p1[0]][p1[1]]=True
  flag=False
  for i in range(4):
    a_n=p1[0]+dir[i][0]
    b_n=p1[1]+dir[i][1]
    if(0<=a_n and a_n<n and 0<=b_n and b_n<n):
      if connect_check([a_n,b_n],p2,n,visited):
        flag=True

  return flag


def DFS(dir_idx,a,b,n,end_x,p,table,visited,bored):
  # bored는 list안에 넣어서 call by reference
  if not visited[a][b]:
    visited[a][b]=True
    t_cnt=0
    for i in range(4):
      a_n=a+dir[i][0]
      b_n=b+dir[i][1]
      if(0<=a_n and a_n<n and 0<=b_n and b_n<n):
        if table[a_n][b_n]:
          t_cnt+=1
    
    flag=True
    a_f=a+dir[dir_idx][0]
    b_f=b+dir[dir_idx][1]
    if(0<=a_f and a_f<n and 0<=b_f and b_f<n):
      a_n=a+dir[(dir_idx+1)%4][0]
      b_n=b+dir[(dir_idx+1)%4][1]
      if(0<=a_n and a_n<n and 0<=b_n and b_n<n):
        table_cp=copy.deepcopy(table)
        table_cp[a][b]=True
        if not connect_check([a_f,b_f],[a_n,b_n],n,table_cp):
          flag=False
      a_n=a+dir[(dir_idx+3)%4][0]
      b_n=b+dir[(dir_idx+3)%4][1]
      if(0<=a_n and a_n<n and 0<=b_n and b_n<n):
        table_cp=copy.deepcopy(table)
        table_cp[a][b]=True
        if not connect_check([a_f,b_f],[a_n,b_n],n,table_cp):
          flag=False

    if(t_cnt<=1 and flag):
      table[a][b]=True
      dir_idxs=[0,0,0]
      if (random.random()<p):
        dir_idxs[0]=dir_idx
        if (random.random()<0.5):
          dir_idxs[1]=(dir_idx+1)%4
          dir_idxs[2]=(dir_idx+3)%4
        else:
          dir_idxs[1]=(dir_idx+3)%4
          dir_idxs[2]=(dir_idx+1)%4
      else:
        dir_idxs[2]=dir_idx
        if (random.random()<0.5):
          dir_idxs[0]=(dir_idx+1)%4
          dir_idxs[1]=(dir_idx+3)%4
        else:
          dir_idxs[0]=(dir_idx+3)%4
          dir_idxs[1]=(dir_idx+1)%4

      for idx in dir_idxs:
        a_n=a+dir[idx][0]
        b_n=b+dir[idx][1]
        if(0<=a_n and a_n<n and 0<=b_n and b_n<n):
          DFS(idx,a_n,b_n,n,end_x,p,table,visited,bored)
    
    elif((not bored[0]) and ((a==n-1 and b==end_x-1)or(a==n-1 and b==end_x+1)or (a==n-2 and b==end_x))):
      table[a][b]=True
      bored[0]=True
    
    elif(a==n-2 and (b==end_x-1 or b==end_x+1) and (not bored[0])):
      table[a][b]=True
      if (random.random()<0.5):
        DFS(2,n-1,b,n,end_x,p,table,visited,bored)
      else:
        DFS(1,n-2,end_x,n,end_x,p,table,visited,bored)


def build_maze(n,p,start_x,end_x,table):
  table.clear() #size도 사라지고, id값 바뀌지 않음.
  visited=[]
  bored=[False]

  for i in range(n):
    table.append([])
    visited.append([])
    for j in range(n):
      table[i].append(False)
      visited[i].append(False)
  table[n-1][end_x]=True
  visited[n-1][end_x]=True
  DFS(2,0,start_x,n,end_x,p,table,visited,bored)
  return bored[0]


def DFS_path(dir_idx, pos, n, end_x, table, path, visited, cnt):
  # cnt는 list객체
  a=pos[0]
  b=pos[1]
  if a==n-1 and b==end_x:
    path[a][b]=True
    return True
  if not visited[a][b]:
    visited[a][b]=True
    path[a][b]=True
    for i in range(4):
      a_n=a+dir[i][0]
      b_n=b+dir[i][1]
      if (0<=a_n and a_n<n and 0<=b_n and b_n<n):
        flag=False
        if table[a_n][b_n]:
          flag=DFS_path(i, [a_n,b_n], n, end_x, table, path, visited, cnt)
          if flag:
            if dir_idx!=i:
              cnt[0]+=1
            return True
    # 4가지 방향 모두 끝에 도달하지 못했을 경우
    path[a][b]=False
  return False


def build_path(n, start_x, end_x, table, path):
  cnt=[0]
  path.clear()
  visited=[]
  for i in range(n):
    path.append([])
    visited.append([])
    for j in range(n):
      path[i].append(False)
      visited[i].append(False)
  DFS_path(2,[0,start_x],n,end_x,table,path,visited,cnt)
  return cnt[0]


def print_maze(table, start_x, end_x):
  n=len(table)
  W="  "
  B="##"
  print(B*(start_x+1),end='')
  print(W,end='')
  print(B*(n-start_x))

  for i in range(n):
    print(B,end='')
    for j in range(n):
      if table[i][j]:
        print(W,end='')
      else:
        print(B,end='')
    print(B)
  
  print(B*(end_x+1),end='')
  print(W,end='')
  print(B*(n-end_x))
  print()


def print_path(table, path, start_x, end_x):
  n=len(table)
  W="  "
  B="##"
  P="''"
  print(B*(start_x+1),end='')
  print(P,end='')
  print(B*(n-start_x))

  for i in range(n):
    print(B,end='')
    for j in range(n):
      if table[i][j]:
        if path[i][j]:
          print(P,end='')
        else:
          print(W,end='')
      else:
        print(B,end='')
    print(B)

  print(B*(end_x+1),end='')
  print(P,end='')
  print(B*(n-end_x))
  print()


def show_maze(table, start_x, end_x):
  n=len(table)
  W=[255,255,255]
  B=[0,0,0]
  im=[]
  im.append([])
  [im[0].append(B) for i in range(start_x+1)]
  im[0].append(W)
  [im[0].append(B) for i in range(n-start_x)]
  for i in range(n):
    im.append([B])
    for j in range(n):
      if table[i][j]:
          im[i+1].append(W)
      else:
        im[i+1].append(B)
    im[i+1].append(B)
  im.append([])
  [im[n+1].append(B) for i in range(end_x+1)]
  im[n+1].append(W)
  [im[n+1].append(B) for i in range(n-end_x)]

  return im


def show_path(table, path, start_x, end_x):
  n=len(table)
  W=[255,255,255]
  B=[0,0,0]
  P=[255,0,0]
  im=[]
  im.append([])
  [im[0].append(B) for i in range(start_x+1)]
  im[0].append(P)
  [im[0].append(B) for i in range(n-start_x)]
  for i in range(n):
    im.append([B])
    for j in range(n):
      if table[i][j]:
        if path[i][j]:
          im[i+1].append(P)
        else:
          im[i+1].append(W)
      else:
        im[i+1].append(B)
    im[i+1].append(B)
  im.append([])
  [im[n+1].append(B) for i in range(end_x+1)]
  im[n+1].append(P)
  [im[n+1].append(B) for i in range(n-end_x)]

  return im


def build_edge(im_m,im_p,table,path,start_x,end_x):
  n=len(table)
  im_m.append([])
  im_p.append([])
  [im_m[0].append(False) for i in range(start_x+1)]
  [im_p[0].append(False) for i in range(start_x+1)]
  im_m[0].append(True)
  im_p[0].append(True)
  [im_m[0].append(False) for i in range(n-start_x)]
  [im_p[0].append(False) for i in range(n-start_x)]
  for i in range(n):
    im_m.append([False])
    im_p.append([False])
    for j in range(n):
      if table[i][j]:
        im_m[i+1].append(True)
        if path[i][j]:
          im_p[i+1].append(True)
        else:
          im_p[i+1].append(False)
      else:
        im_m[i+1].append(False)
        im_p[i+1].append(False)
    im_m[i+1].append(False)
    im_p[i+1].append(False)
  im_m.append([])
  im_p.append([])
  [im_m[n+1].append(False) for i in range(end_x+1)]
  [im_p[n+1].append(False) for i in range(end_x+1)]
  im_m[n+1].append(True)
  im_p[n+1].append(True)
  [im_m[n+1].append(False) for i in range(n-end_x)]
  [im_p[n+1].append(False) for i in range(n-end_x)]
  return im_m, im_p


if __name__=="__main__":
    # main
    table=[]
    path=[]
    n=8

    for i in tqdm.tqdm(range(10)):
        p=0.3+(0.7/10)*i
        fig=plt.figure(figsize=(16,4))
        fig.suptitle('P='+str(p), fontsize=16)
        for t in range(8):
            start_x=random.randrange(0,n)
            end_x=random.randrange(0,n)
            build_maze(n,p,start_x,end_x,table)
            cnt=build_path(n,start_x,end_x,table,path)
        
            im_m=show_maze(table,start_x,end_x)
            im_p=show_path(table,path,start_x,end_x)
            plt.subplot(2,8,t+1)
            plt.imshow(np.array(im_m))
            plt.title('cnt='+str(cnt))
            plt.subplot(2,8,t+1+8)
            plt.imshow(np.array(im_p))
        plt.savefig(PATH+"examples/straightness="+str(i)+".png")




