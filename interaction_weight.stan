
data {
  int<lower=1> N;       // number of observations
  vector[N] log_gest;   
  vector[N] log_weight;
  vector[N] z;
  vector[N] z_log_gest;
  
}
parameters {
  vector[4] beta;           // coefs
  real<lower=0> sigma;  // error sd for Gaussian likelihood
}
model {
  // Log-likelihood
  target += normal_lpdf(log_weight | beta[1] + beta[2] * log_gest + beta[3] * z + beta[4] * z_log_gest , sigma);

  // Log-priors
  target += normal_lpdf(sigma | 0, 1)
          + normal_lpdf(beta | 0, 1);
}
