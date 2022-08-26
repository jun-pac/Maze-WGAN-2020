# -*- coding: utf-8 -*-
# python3 maze_dataset_Gen.py --size 16 --num 1000 --p 70 92 --name M --show_sample False
# Dataset ����� �ڵ�
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

# �̸� ������ M _ (n) _ (Number of data/1000) + (�ʿ��ϴٸ� cnt) + (�ʿ��ϴٸ� p)
# (train/test) �̰� �̸� ������ �� �ʿ䰡 ���� ������ �����ϴ�.
# [[map1(2-dimension),path1(2-dimension)],[map2,path2],...] �̷� ������ �����.
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

# �ڵ��� �غ��� numpy array�� �����ϴ� �ǹ̰� ���� ����, cnt���� ���� ��� �����ؾ� ���� �ʳ� �ʹ�.
# �ϴ� ���� ���ῡ �� ���� cnt�� �����ִ� �Լ��� �ϼ��ϰ�,
# cnt���� ���� accuracy�� ���ǹ��� �������� Ȯ���� ���� ���̴�.
# accuracy�� ���� ���� 0.5�� threshold������ block���� ������ ����, ��� path�� ������ Ȯ���ϴ� ���.
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
  # turn_cnt, branch_cnt, length�� ���غ���.
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