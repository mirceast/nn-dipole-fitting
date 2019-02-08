function generate_dipole_dataset(sa,n_dipoles,n_runs,snr,train_size,valid_size,test_size,save_name)

% Useful variables
n_chan = size(sa.locs_2D,1);
max_location = max(sa.grid_fine);
min_location = min(sa.grid_fine);
max_moment = 10 * ones(1,3);
min_moment = -max_moment;

% Generate train set
dipoles_train = nan(n_dipoles,6,train_size);
field_train = nan(n_chan,train_size);
parfor i = 1:train_size
    % Generate dipoles
    [dipoles_train(:,:,i),~,field_train(:,i)] = generate_random_dipoles(n_dipoles,sa,snr);
end

% Generate validation set
dipoles_valid = nan(n_dipoles,6,valid_size);
field_valid = nan(n_chan,valid_size);
parfor i = 1:valid_size
    % Generate dipoles
    [dipoles_valid(:,:,i),~,field_valid(:,i)] = generate_random_dipoles(n_dipoles,sa,snr);
end

% Generate test set
dipoles_test = nan(n_dipoles,6,test_size);
estimated_dipoles_test = nan(n_dipoles,6,test_size);
field_test = nan(n_chan,test_size);
parfor i = 1:test_size
    % Generate dipoles
    [dipoles_test(:,:,i),~,field_test(:,i)] = generate_random_dipoles(n_dipoles,sa,snr);
    % Find dipoles - do multiple runs
    [estimated_dipoles_test(:,:,i),estimated_field] = fit_dipoles(n_dipoles,field_test(:,i),sa,n_runs);

end

save(save_name,'sa','max_location','min_location','max_moment','min_moment','snr','n_runs','dipoles_train','field_train','dipoles_valid','field_valid','dipoles_test','field_test','estimated_dipoles_test','-v7.3')


