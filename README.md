# nn-dipole-fitting
Using neural networks for EEG dipole fitting

# Description
Single dipole fitting from EEG/MEG data is traditionally performed using the [Levenberg–Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm). While being relatively fast on modern hardware, the LM algorithm is not appropriate for certain scenarios such as dipole fitting in a moving window within long EEG recordings. The present approach generates a million random dipoles and calculates the EEG surface potential in Matlab, using a realistic volume conductor model. A feedforward neural network is then trained in PyTorch to predict the dipole location given the surface EEG recordings. Training is guided by an independent dataset containing 100K dipole-field pairs. The network is then evaluated on yet another set of 100K dipole-field pairs. The present approach is shown to outperform the traditional LM algorithm, especially when recordings are very noisy. Of particular interest is that as source depth increases, the performance advantage of the current approach widens, opening up the possibility of studying deep brain structures with EEG.

![Reults](https://raw.githubusercontent.com/mirceast/nn-dipole-fitting/master/Results.png)
