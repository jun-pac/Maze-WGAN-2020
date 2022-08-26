# Maze-WGAN-2020
Solving generated maze with WGAN

## Maze generator
This is a stochastic maze generator using DFS. 
The generated maze is guaranteed to have only one path. 
The parameter p(Straightness) specifies how likely it is to go straight in the maze creation DFS process.<br/>
Enter the following command to see how the created maze changes according to straightness.
```
python mazeGen.py
```
![straightness=0](https://user-images.githubusercontent.com/100084401/186951037-b520e082-fcd2-4123-8480-f9d2f770f923.png)
![straightness=9](https://user-images.githubusercontent.com/100084401/186951300-46299166-6089-486d-afb0-a72ab22ef7f9.png)

Note that it's not essential command to create dataset.


## Maze dataset generator
It combines input and label pairs to create a dataset. You can specify the size of the maze, the number of data, the range of straightness (two values in [0,100]), and the name of the dataset.
The saved dataset is a .pickle file in which maze pairs are stored as numpy arrays, not in image.
```
python maze_dataset_Gen.py --size 16 --num 1000 --p 70 92 --name M --show_sample False
```
If the show_sample is set to True, the first pair of the dataset is saved as an image.
<p align="center">
  <img src="https://user-images.githubusercontent.com/100084401/186952390-2bc8b152-92fa-49f4-8405-817581acb86e.png" width="600"/>
</p>

