% clear all
clearvars -except sa mri
close all
home

if ~exist('sa','var')
    load('METH_Biosemi64_sa.mat','sa')
    load mri
end

data_folder = 'C:\Users\Mircea\Google Drive\Data\nn-dipole-fitting/Multi_Dipole/';
% data_folder = '/home/mstoica/Data/Multi_Dipole/';
if ~exist(data_folder,'dir')
    mkdir(data_folder)
end

% Sample locations from grid_fine
% Moment between 0 and 10
% Use mk_sensors_plane to convert 3d sensor locations to 2d
% Maybe do the 2d interpolation and use that image for the convolution

% Define stuff
n_dipoles = [1,200];
train_size = 1e6;
valid_size = 1e5;

generate_clean_dipole_dataset(sa,n_dipoles,train_size,valid_size,[data_folder,'Clean Dataset.mat'])

















