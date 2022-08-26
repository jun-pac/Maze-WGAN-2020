# -*- coding: utf-8 -*-
# python3 maze_dataset_Gen.py --size 16 --num 1000 --p 70 92 --name M --show_sample False
# Dataset 만드는 코드
import pickle
import time
from argparse import ArgumentParser
from mazeGen import *
sys.setrecursionlimit(10000)

PATH=os.getcwd()
PATH=PATH+'/dataset/'

parser = ArgumentParser()

parser.add_argument(
    "--size",
    default=16,
    type=int,
    help="Diameter of maze",
)
parser.add_argument(
    "--num",
    default=1000,
    type=int,
    help="Size of dataset",
)
parser.add_argument(
    "--p",
    nargs="+",
    default=[70,92],
    type=int,
    help="Straightness of generated maze",
)
parser.add_argument(
    "--name",
    default="M",
    type=str,
    help="Identifier of dataset",
)
parser.add_argument(
    "--show_sample",
    default=True,
    type=bool,
    help="Generate sample image",
)

args = parser.parse_args()

# 이름 형식은 M _ (n) _ (Number of data/1000) + (필요하다면 cnt) + (필요하다면 p)
# (train/test) 이건 미리 구분해 둘 필요가 없어 보여서 뺐습니다.
# [[map1(2-dimension),path1(2-dimension)],[map2,path2],...] 이런 순서로 저장됨.
num=args.num
n=args.size
name=args.name+'_'+str(16)+'_'+str(num//1000)
p1,p2=args.p

table=[]
path=[]
bucket=[]
cnt_list=[]
t1=time.time()
for i in tqdm.tqdm(range(num)):
  p=random.randrange(p1,p2)/100
  start_x=random.randrange(0,n)
  end_x=random.randrange(0,n)
  build_maze(n,p,start_x,end_x,table)
  cnt=build_path(n,start_x,end_x,table,path)
  cnt_list.append(cnt)
  im_m=[]
  im_p=[]
  build_edge(im_m,im_p,table,path,start_x,end_x)
  bucket.append([[],[]])
  bucket[i][0]=copy.deepcopy(im_m)
  bucket[i][1]=copy.deepcopy(im_p)
  #if i%1000==999:
  #  print(i+1,"/",num,"finished :",time.time()-t1)

with open(PATH+name+'.pickle', 'wb') as f:
    pickle.dump(np.array(bucket), f, pickle.HIGHEST_PROTOCOL)
print("finished. time :",time.time()-t1)

# M_8_200 finished. time : 1781.6787614822388
# M_16_60 finished. time : 7721.626291036606 (p=0.7~0.95)
# M_16_30 finished. time : 4266.527472019196
# M_16_10 finished. time : 1662.545743227005

# 코딩을 해보니 numpy array로 저장하는 의미가 거의 없고, cnt까지 같이 묶어서 저장해야 하지 않나 싶다.
# 일단 오늘 저녁에 할 일은 cnt를 세어주는 함수를 완성하고,
# cnt수에 따른 accuracy가 유의미한 수준인지 확인해 보는 것이다.
# accuracy를 구할 때는 0.5를 threshold값으로 block들을 구분한 다음, 모두 path와 같은지 확인하는 방법.
dir=[[-1,0],[0,1],[1,0],[0,-1]]

def DFS_atr(table,path,visited,N,pos,dir_idx,end_x,turn_cnt,branch_cnt,length):
  if visited[pos[0]][pos[1]]:
    return False
  visited[pos[0]][pos[1]]=True
  length[0]+=1
  if pos!=[N-1,end_x]:
    for i in range(4):
      a_n=pos[0]+dir[i][0]
      b_n=pos[1]+dir[i][1]
      if (0<=a_n<N and 0<=b_n<N):
        if (path[a_n][b_n] and i!=((dir_idx+2)%4)):
          dir_temp=i
          a_f=a_n
          b_f=b_n
        if (not path[a_n][b_n]) and table[a_n][b_n]:
          branch_cnt[0]+=1
    if dir_temp!=dir_idx:
      turn_cnt[0]+=1
    DFS_atr(table,path,visited,N,[a_f,b_f],dir_temp,end_x,turn_cnt,branch_cnt,length)

def get_attribute(table,path):
  # turn_cnt, branch_cnt, length를 구해보자.
  N=len(table)
  for i in range(N):
    if path[0][i]:
      start_x=i
    if path[N-1][i]:
      end_x=i
  turn_cnt=[0]
  branch_cnt=[0]
  length=[0]
  visited=[[False for i in range(N)] for j in range(N)]
  DFS_atr(table,path,visited,N,[0,start_x],2,end_x,turn_cnt,branch_cnt,length)
  return turn_cnt[0], branch_cnt[0], length[0]


m=0
t,b,l=get_attribute(bucket[m][0],bucket[m][1])
print(t,b,l)
fig=plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(bucket[m][0])
plt.subplot(1,2,2)
plt.imshow(bucket[m][1])
plt.savefig(PATH+"sample_data.png")