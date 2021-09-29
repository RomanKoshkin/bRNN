# Self-organization in a recurrent network of binary neurons &mdash; Single-core C++/Python implementation

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https:/travis-ci.org/joemccann/dillinger.svg?branch=master)

![Teaser image](https://svgshare.com/i/acV.svg =250x250)

## Release notes

This repository is a single-core C++/Python implementation of a recurrent network of binary neurons that given the right parameters, self-organizes to a fixed number of cell-assemblies, focusing on ease-of-use, performance and reproducibility and speed.

**Ease of use**
* Supports in-simulation parameter manipulation.
* Easy-to-use Python API

**Performance**
* Performance-critical code written in C++

**Reproducibility and Speed**
* Saves model states, model states can be loaded at any time, which speeds up experimentation.


## Requirements

* Linux (other OSs are not tested)
* Python 3.9
* Python libraries: `pip install scikit-learn==0.23.2 scikit-network==0.20.0 scipy==1.6.1 numpy==1.19.2 networkx==2.5 tqdm matplotlib==3.3.4`.


## Getting started

First compile the C++ module:

```.bash
g++ -std=gnu++11 -Ofast -shared -fPIC -ftree-vectorize -march=native -mavx bmm_9_haga_grid.cpp -o
```

## Running a grid of simulations on a cluster

```.bash
sh grid_search.sh
```
This runs a grid of 10000 simulations. You can control the cells that will run in the `grid_serach.sh` script.

Outputs from the each cell of the grid (that is, each instantiation of the model) will be saved to the path set in the variable `path_to_save_wts_on_bucket` in the file `grid_cell.py`. Make sure to set other paths for data dumps (`path_to_save_wts_on_flash`, `slurm_out_folder`) as necessary.



## Running evolved or naive networks from Jupyter

See `Examples.ipynb`.


## License

Copyright &copy; 2021, Okinawa Institute of Science and Technology.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## Development

This is a research reference implementation and is treated as a one-time code drop. As such, we do not accept outside code contributions in the form of pull requests.

## Acknowledgements

Big thanks to Naoki Hiratani for his support.