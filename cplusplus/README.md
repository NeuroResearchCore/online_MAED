# Online MAED

Adapting https://github.com/NeuroResearchCore/online_MAED to C++ using [Armadillo](http://arma.sourceforge.net/).

## Setup
The dependencies are C++ 17 and Armadillo.

On MacoS, `brew install armadillo`.

On Linux and Windows systems, follow the steps on the Armadillo website.

There is a `Makefile` that works on MacOS when you run
```
make result
./result
```

Currently, `./result` will print out the representative features and their respective labels. Along with the starting and ending indices for each batch.

The entry point is `main.cpp`. Currently uses a dummy dataset to run on.

## Example
There are no unit tests, but `example.cpp` is there to print out the results of some utiility functions. To run it:
```
make example
./example
```
