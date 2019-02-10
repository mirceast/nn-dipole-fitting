function generate_and_fit_dipole_dataset(sa,n_dipoles,n_runs,snr,test_size,save_name)

% Useful variables
n_chan = size(sa.locs_2D,1);
max_location = max(sa.grid_fine);
min_location = min(sa.grid_fine);
max_moment = 10 * ones(1,3);
min_moment = -max_moment;

% Generate test set
dipoles_test = nan(n_dipoles,6,test_size);
estimated_dipoles_test = nan(n_dipoles,6,test_size);
field_test = nan(n_chan,test_size);
res_out = nan(test_size,1);
parfor i = 1:test_size
    % Generate dipoles
    [dipoles_test(:,:,i),~,field_test(:,i)] = generate_random_dipoles(n_dipoles,sa,snr);
    % Find dipoles - do multiple runs
    [estimated_dipoles_test(:,:,i),estimated_field,res_out(i)] = fit_dipoles(n_dipoles,field_test(:,i),sa,n_runs);
end
chan_locs = sa.locs_2D;
save(save_name,'sa','max_location','min_location','max_moment','min_moment','snr','n_runs','dipoles_test','field_test','estimated_dipoles_test','res_out','chan_locs','-v7.3')


