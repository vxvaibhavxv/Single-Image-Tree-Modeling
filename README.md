# Single Image Tree Modelling

An application that accepts an image of a tree (only `.jpg`) along with two descriptive strokes, one for the crown and another for the trunk. Using this information, it builds a 3D model for the tree.

## How to run?

```shell
mkdir build
cd build
cmake ..
make
cd ..
./Project <tree-image>.jpg
```

## Examples

Input              | Output
:-----------------:|:-----------------------:
![](readme/1.png)  |  ![](readme/1-out.png)
![](readme/2.png)  |  ![](readme/2-out.png)
![](readme/3.jpg)  |  ![](readme/3-out.png)