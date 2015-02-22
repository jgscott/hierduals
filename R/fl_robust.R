library(Rcpp)
library(glmgen)
library(foreach)
sourceCpp('../cpp/fl_dp.cpp')


# Simulate some data
N = 250
x = seq(0, 1, length=N)
fx = rep(-2, N)
fx[51:100] = 2
sigma = 1
y = fx + sigma*rt(N, 3)

# Solution path for 1D fused lasso
system.time(model2 <- genlasso::fusedlasso1d(y))
AIC_path = 2*model2$df + summary(model2)[,3]
jbest = which.min(AIC_path)
beta_hat0 = model2$beta[,jbest]

plot(x, y, pch=19, col=rgb(0.2,0.2,0.2,0.8), bty='n', las=1, 
	main='')
lines(x, fx, col='blue', lwd=2, type='s')
lines(x, beta_hat0, lwd=2, type='s', col='darkgrey', lty='dotted')

# Calculate MSE
mean((beta_hat0 - fx)^2)


# Now with Huber loss
lambda_grid = 10^seq(-2, 2, length=100)
system.time({
aic_grid = foreach(lambda=lambda_grid, .combine='c') %do% {
	beta_it = rep(mean(y), N)
	for(i in 1:50) {
		res_it = y - beta_it
		# Update variational parameters
		u = res_it - sign(res_it)
		u[abs(res_it) < 1] = 0
		# Update beta
		beta_it = fl_dp(y-u, lambda)
	}
	thisdf = sum(abs(diff(beta_it)) > 1e-6)
	thisAIC = sum(huberpen(res_it, 1)) + 2*thisdf
	thisAIC
}
})
plot(aic_grid)
lambda_best = lambda_grid[which.min(aic_grid)]

# Fit again at the best choice of lambda,
# since we were silly and didn't save the answers
beta_it = rep(mean(y), N)
for(i in 1:50) {
	res_it = y - beta_it
	# Update variational parameters
		u = res_it - sign(res_it)
	u[abs(res_it) < 1] = 0
	# Update beta
	beta_it = fl_dp(y-u, lambda_best)
}

# Compare the fits
plot(x, y, pch=19, col=rgb(0.2,0.2,0.2,0.3), bty='n', las=1, 
	main='')
lines(x, fx, col='red', lwd=2, type='s', lty='dotted')
lines(x, beta_hat0, lwd=2, type='s', col='darkgrey', lty='solid')
lines(x, beta_it, lwd=2, type='s', col='blue', lty='dashed')

legend('bottomright', legend=c('Truth', 'Fused lasso (MSE = 1.45)', 'Robust fused lasso (MSE = 0.08)'),
	col=c('red', 'darkgrey', 'blue'), lty=c('dotted', 'solid', 'dashed'), bg='white', lwd=2)

# Calculate MSE
mean((beta_it - fx)^2)



# Visualizing the Huber loss

# Huber penalty
cppFunction('
	NumericVector huberpen(NumericVector x, double kink) {
	  int d = x.size();
	  double k2 = kink/2.0;
	  double absx;
	  NumericVector result(d);
	  for(int i=0; i<d; i++) {
	    absx = fabs(x[i]);
	    result[i] = ( (absx) <= (kink) ? (0.5*x[i]*x[i]) : (kink*(absx - k2)) );
	  }
	  return result;
	}
')


curve(huberpen(x, 1), from=-3, to=3)
abline(-1/2,1)
