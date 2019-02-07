function [dipoles,dipole_field,noisy_field] = generate_random_dipoles(n_dipoles,sa,snr)

% Useful variables
n_chan = size(sa.locs_2D,1);

% Generate random dipoles, choose random locations from the fine grid
dipoles = zeros(n_dipoles,6);
dipoles(:,1:3) = sa.grid_fine(unidrnd(length(sa.grid_fine),[n_dipoles,1]),:);
dipoles(:,4:6) = 20 * rand(n_dipoles,3) - 10;

% Forward propagation
dipole_field = forward_general(dipoles,sa.fp);

% Sum up all dipole fields                                
dipole_field = sum(dipole_field,2);

% Add some noise
noise = randn(n_chan,1);
noisy_field = snr * dipole_field/norm(dipole_field) + noise/norm(noise); % noise is added with SNR=snr 
