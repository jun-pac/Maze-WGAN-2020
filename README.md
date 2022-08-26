# Maze-WGAN-2020
Solving the maze is a complex inference problem that cannot be solved by simple classification or regression. I was wondering if it is possible to solve the maze using GAN. Discriminate whether the output heatmap is the correct path or not would be an easier task than generating the correct path, so I thought that the generator could get more clues than using only the MSE loss. I used the CGAN that discriminator receives input and label as a pair, and wesserstein loss was used to alleviate mode collapse.<br/>
The objective of this toy project is to see how well WGAN solves the generated maze.

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

## WGAN maze solver
We checked whether the maze can be solved by training a WGAN based regression model. The generator was trained to generate a heatmap close to the label(boolean array indicating correct path), and weight-clipping discriminator takes this heatmap as an input to discriminate it. Both MSE loss and adversarial loss were used to train the generator, and the MSE loss was multiplied by 20 times.<br/>
Learning was conducted in jupyter notebook in colab environment. The following command doesn't work right now, but I'll fix it later to make it work:
```
python WGAN.py
```
The output is the boolean heatmap with a threshold of 0.5, and it is defined as solving the maze if and only if all pixels of the output are the same as all pixels of the label. **Accuracy** is defined as the number of mazes solved out of the total number of mazes.

## Results
Using the following command, a total of 60,000 datas were created and used for learning.
```
python maze_dataset_Gen.py --size 16 --num 60000 --p 70 92 --name M --show_sample False
```
It was trained for 10 epochs, and the batch size was 100. The final MSE loss was 6.13e-3, and the accuracy was 0.991.<br/>
Which means that almost all mazes have been perfectly solved! The following are four randomly selected heatmaps and their corresponding labels.<br/>
<p align="center">
  <img src="https://user-images.githubusercontent.com/100084401/186957857-f1906ce2-b003-4b3d-bad3-d7934fa1bdb3.png" width="600"/>
</p>

The following shows *all the mazes* that the WGAN maze solver could not solve among 60000 mazes.<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/100084401/186958122-d8a443e8-c86e-47f8-97a9-c2393f271157.png" width="600"/>
</p>
It seems that more research is needed on why and which mazes cannot be solved, and what the limits of the WGAN maze solver are.

## Reference
[1] Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." Advances in neural information processing systems 30 (2017).
