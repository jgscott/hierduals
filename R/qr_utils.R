quantreg_trendfilter = function(y, q=0.5, k=3L, lambda = 1, rel_tol = 1e-4, max_it = 100) {
# fits a nonparametric quantile regression model via polynomial trend filtering
# y: ordered data points, assumed to be on a regularly spaced grid
# q in (0,1): the desired conditional quantile

	# package dependencies
	require(glmgen)
	
	# initialize
	N = length(y)
	
	# Initial guess
	beta_hat = rep(mean(y), N)
	residual = y - beta_hat
	kappa = 1.0 - 2*q
	
	# Cheap and easy approach
	# For a "real" version we really should be checking convergence
	converged = FALSE
	travel = max(1, 2*rel_tol)
	residual = y - beta_hat
	it_counter = 0
	while(!converged && it_counter <= max_it) {
		# Compute the weights and pseudo-data in Taylor approximation of log likelihood
		weights = sign(residual)/(residual) + 1e-6
		z = y - kappa/weights
		tfk = glmgen::trendfilter(z, weights=weights, k = k, family='gaussian', lambda=lambda)
		beta_hat = tfk@beta
		new_residual = drop(y - beta_hat)
		travel = sum( (residual - new_residual)^2 )
		residual = new_residual
		converged = travel/(sum(new_residual^2) + rel_tol) < rel_tol
		it_counter = it_counter + 1
	}
	list(beta = beta_hat, tfk=tfk)
}

