# FYS-STK Project 3  
**Solving Partial Differential Equations with Neural Networks**

Authors: Jenny Guldvog and Ingvild Olden Bjerkelund

This repository contains the code and material developed for **Project 3** in the course  
**FYS-STK4155 – Applied Data Analysis and Machine Learning** at the University of Oslo.

The project follows the official project description:
https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/project3.html

---

## Project Overview

The aim of this project is to investigate how neural networks can be used to solve
partial differential equations (PDEs), and to compare this approach with a classical
numerical method. As a case study, we consider the one-dimensional heat (diffusion)
equation with homogeneous Dirichlet boundary conditions and a known analytical solution.

---

## Project Components

The main components of the project are:

- Formulation of the physical problem and derivation of the analytical solution
- Numerical solution of the PDE using a finite-difference scheme (Forward-Time Central-Space, FTCS)
- Solution of the PDE using a neural network, PINN, where the governing equation and boundary
  conditions are incorporated into the loss function
- Comparison of accuracy, convergence behaviour, and computational efficiency between
  the two approaches

---

## Repository Structure

```
FYS-STK-Project-3/
.
├── main/ # Main jupyter notebooks for implementations and figures
├── src/ # Source code and helper modules
├── FYSSTK_P3.pdf # Project report (PDF)
├── environment.yml # Conda environment specification
├── requirements.txt # Python dependencies
└── README.md # Repository documentation
```

---

## Running the Code

The project is implemented in Python. All required dependencies are listed in
`requirements.txt` and `environment.yml`.

To reproduce the results, run the notebooks in main/.

---

## Course Context

This work is carried out as part of the compulsory coursework in **FYS-STK4155** at the
University of Oslo and follows the guidelines given in  
*Project 3 – Solving Partial Differential Equations with Neural Networks*.
