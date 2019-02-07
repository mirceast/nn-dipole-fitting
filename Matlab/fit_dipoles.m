function [estimated_dipoles,estimated_field] = fit_dipoles(n_dipoles,input_field,sa,n_runs)

n_chan = size(sa.locs_2D,1);
res_out = nan(n_runs,1);
estimated_dipoles = nan(n_dipoles,6,n_runs);
estimated_field = nan(n_chan,n_runs);
for i = 1:n_runs
    [estimated_dipoles(:,:,i),res_out(i),~,estimated_field(:,i)] = dipole_fit_field(input_field,sa.fp,n_dipoles); %make the fit 
end
estimated_dipoles = estimated_dipoles(:,:,argmin(res_out));
estimated_field = estimated_field(:,argmin(res_out));
