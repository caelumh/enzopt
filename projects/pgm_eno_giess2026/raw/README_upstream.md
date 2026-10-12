[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EnzymeML/Giess_2026/HEAD)

# This repository accompanies the paper Giess 2026

> "FAIR NMR: acquisition, processing, and modelling of time-resolved NMR data according to the FAIR principles"  

The folder structure follows the structure of the paper.  
Here you can find all data and Jupyter Notebooks.

## Using the Jupyter Notebooks

### With Binder

You can use Binder to run the Jupyter Notebooks online, by clicking on this badge:  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EnzymeML/Giess_2026/HEAD)  
(NOTE: you will be directed to Binder, this will not open a new tab! You can use right-click -> Open in new Tab)  
  
This will start a JupyterLab session in your browser.  
On the left you find the folder structure. Double-click on the folder you want to open, e.g. Scenario2. The Jupyter Notebooks have a small orange image icon. Double-click on the notebook you want to run.  
Now you can interact with the notebook. To run all cells click the icon with the two arrows, left of Download, and click on Restart in the popup.  
Each cell will automatically run from top to bottom. When a code cell was run a number appears in the brackets on the left.  

### Locally

To run the Jupyter Notebooks locally, you need to have Python installed on your computer.

You can install the required packages using pip. Open a terminal and navigate to the folder where you have cloned this repository, then run:

```bash
pip install -r requirements.txt
```

After installing the required packages, you can start JupyterLab by running:

```bash
jupyter lab
```

This will open JupyterLab in your default web browser. You can then navigate to the folder structure on the left and open the Jupyter Notebooks as described in the Binder section.
