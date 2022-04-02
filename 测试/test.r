a <- 7      # Lower threshold
d <- 11     # Dimensions
n <- 1e5    # Sample size
sigma <- 3  # Original SD
#
# The algorithm.
#
set.seed(17)

# 计算累计卡方密度
u.max <- pchisq((a/sigma)^2, d, lower.tail=FALSE)
if (u.max == 0) stop("The threshold is too large.")
u <- runif(n, 0, u.max)  # 均匀分布采样
rho <- sigma * sqrt(qchisq(u, d, lower.tail=FALSE)) 
x <- matrix(rnorm(n*d, 0, 1), ncol=d)  # rnorm(n, mean = 0, sd = 1) 
y <- t(x * rho / apply(x, 1, function(y) sqrt(sum(y*y))))
#
# Draw histograms of the marginal distributions.
#
h <- function(z) {
  s <- sd(z)
  hist(z, freq=FALSE, ylim=c(0, 1/sqrt(2*pi*s^2)),
       main="Marginal Histogram",
       sub="Best Normal Fit Superimposed")
  curve(dnorm(x, mean(z), s), add=TRUE, lwd=2, col="Red")
}
par(mfrow=c(1, min(d, 4)))
invisible(apply(y, 1, h))
#
# Draw a nice histogram of the distances.
#
#plot(y[1,], y[2,], pch=16, cex=1/2, col="#00000010") # For figure 2
rho.max <- min(qchisq(1 - 0.001*pchisq(a/sigma, d, lower.tail=FALSE), d)*sigma, 
               max(rho), na.rm=TRUE)
k <- ceiling(rho.max/a)
hist(rho, freq=FALSE, xlim=c(0, rho.max),  
     breaks=seq(0, max(rho)+a, by=a/ceiling(50/k)))
#
# Superimpose the theoretical distribution.
#
dchi <- function(x, d) {
  exp((d-1)*log(x) + (1-d/2)*log(2) - x^2/2 - lgamma(d/2))
}
curve((x >= a)*dchi(x/sigma, d) / (1-pchisq((a/sigma)^2, d))/sigma, add=TRUE, 
      lwd=2, col="Red", n=257)