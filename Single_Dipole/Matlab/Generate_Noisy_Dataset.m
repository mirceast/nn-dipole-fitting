% clear all
clearvars -except sa mri
close all
home

if ~exist('sa','var')
    load('METH_Biosemi64_sa.mat','sa')
    load mri
end

data_folder = 'C:\Users\Mircea\Google Drive\Data\nn-dipole-fitting/';
% data_folder = '/home/mstoica/Data/';
if ~exist(data_folder,'dir')
    mkdir(data_folder)
end

% Sample locations from grid_fine
% Moment between 0 and 10
% Use mk_sensors_plane to convert 3d sensor locations to 2d
% Maybe do the 2d interpolation and use that image for the convolution

% Define stuff
n_dipoles = 1;
n_runs = 5;
snr = [0.1,0.5,1,2,5,10,100];
test_size = 1e4;

for i = 1:numel(snr)
    fprintf('Started SNR %d/%d at %s\n',i,numel(snr),datestr(now));
    generate_and_fit_dipole_dataset(sa,n_dipoles,n_runs,snr(i),test_size,[data_folder,'Dataset SNR ',sprintf('%02d',i),'.mat'])
end

















