function out = normalize_dipole(in,max_location,min_location,max_moment,min_moment)

out = nan(size(in));
out(:,1:3) = (in(:,1:3) - min_location) ./ (max_location - min_location);
out(:,4:6) = (in(:,4:6) - min_moment) ./ (max_moment - min_moment);


