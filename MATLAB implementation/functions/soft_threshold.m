function x = soft_threshold(a,kappa)
x = max(0,a-kappa) - max(0,-a-kappa);
end