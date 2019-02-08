function generate_clean_dipole_dataset(sa,n_dipoles,train_size,save_name)

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
    [dipoles_train(:,:,i),~,field_train(:,i)] = generate_random_dipoles(n_dipoles,sa,[]); % snr = [] which means no noise added
end

save(save_name,'sa','max_location','min_location','max_moment','min_moment','dipoles_train','field_train','-v7.3')


