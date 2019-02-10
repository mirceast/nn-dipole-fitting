function generate_clean_dipole_dataset(sa,n_dipoles,train_size,valid_size,save_name)

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

% Generate valid set
dipoles_valid = nan(n_dipoles,6,valid_size);
field_valid = nan(n_chan,valid_size);
parfor i = 1:valid_size
    % Generate dipoles
    [dipoles_valid(:,:,i),~,field_valid(:,i)] = generate_random_dipoles(n_dipoles,sa,[]); % snr = [] which means no noise added
end

chan_locs = sa.locs_2D;
save(save_name,'sa','max_location','min_location','max_moment','min_moment','dipoles_train','field_train','dipoles_valid','field_valid','chan_locs','-v7.3')


