library(glmgen)
source('qr_utils.R')

# Set up experiment
N = 1000
x = seq(0, 1, length=N)
fx = 5*sin(2*pi*x)

sigma_x = exp(1.5*sin(4*pi*x)) + 0.5
y = fx + rnorm(N, 0, sigma_x)
truefq = qnorm(q, fx, sigma_x)
plot(x, y, pch=19, col=rgb(0.2,0.2,0.2,0.1), bty='n', las=1, 
	main='')
lines(x, truefq, col='red', lwd=2)

# Fit the model for a fixed lambda
q = 0.9
system.time(model3 <- quantreg_trendfilter(y, q=q, k=2L, lambda=1000))

# Plot the result
plot(x, y, pch=19, col=rgb(0.2,0.2,0.2,0.1), bty='n', las=1, 
	main='')
change_ind = which(abs(diff(diff(diff(model3$beta)))) > 5e-5)
rug(x[change_ind], col='black', lwd=3)
lines(x, truefq, col='red', lwd=2)
lines(x, model3$beta, col='blue', lwd=2, lty='dashed')
legend('topright', legend=c('True quantile function (q=0.9)', 'Trend filtering estimate'),
	col=c('red', 'blue'), lty=c('solid', 'dashed'), bg='white')
