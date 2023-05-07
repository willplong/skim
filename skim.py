# %%
import csv
import datetime
import itertools
import os
import pickle
import time

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import vmap
from jax.lib import xla_bridge
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from numpyro.infer import MCMC, NUTS
from sklearn.feature_selection import SelectKBest, VarianceThreshold

# %% [markdown]
# # Define Sparse Regression Model
# 
# We demonstrate how to do (fully Bayesian) sparse linear regression using the approach described in
# [1]. This approach is particularly suitable for situations with many feature dimensions (large P)
# but not too many data points (small N).
# 
# In particular, we consider a quadratic regressor of the form:
# 
# $$f(X) = \text{constant} + \sum_i \theta_i X_i + \sum_{i<j} \theta_{ij} X_i X_j + \text{observation noise}$$
#     
# **References:**
# 1. Raj Agrawal, Jonathan H. Huggins, Brian Trippe, Tamara Broderick (2019), "The Kernel Interaction
#    Trick: Fast Bayesian Discovery of Pairwise Interactions in High Dimensions",
#    (https://arxiv.org/abs/1905.06501)

# %%
def dot(X, Z):
    return jnp.dot(X, Z[..., None])[..., 0]


# The kernel that corresponds to our quadratic regressor.
def kernel(X, Z, eta1, eta2, c, jitter=1.0e-4):
    eta1sq = jnp.square(eta1)
    eta2sq = jnp.square(eta2)
    k1 = 0.5 * eta2sq * jnp.square(1.0 + dot(X, Z))
    k2 = -0.5 * eta2sq * dot(jnp.square(X), jnp.square(Z))
    k3 = (eta1sq - eta2sq) * dot(X, Z)
    k4 = jnp.square(c) - 0.5 * eta2sq
    if X.shape == Z.shape:
        k4 += jitter * jnp.eye(X.shape[0])
    return k1 + k2 + k3 + k4


# Most of the model code is concerned with constructing the sparsity inducing prior.
def model(X, Y, hypers):
    S, P, N = hypers["expected_sparsity"], X.shape[1], X.shape[0]

    sigma = numpyro.sample("sigma", dist.HalfNormal(hypers["alpha3"]))
    phi = sigma * (S / jnp.sqrt(N)) / (P - S)
    eta1 = numpyro.sample("eta1", dist.HalfCauchy(phi))

    msq = numpyro.sample("msq", dist.InverseGamma(hypers["alpha1"], hypers["beta1"]))
    xisq = numpyro.sample("xisq", dist.InverseGamma(hypers["alpha2"], hypers["beta2"]))

    eta2 = jnp.square(eta1) * jnp.sqrt(xisq) / msq

    lam = numpyro.sample("lambda", dist.HalfCauchy(jnp.ones(P)))
    kappa = jnp.sqrt(msq) * lam / jnp.sqrt(msq + jnp.square(eta1 * lam))

    # compute kernel
    kX = kappa * X
    k = kernel(kX, kX, eta1, eta2, hypers["c"]) + sigma**2 * jnp.eye(N)
    assert k.shape == (N, N)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
        obs=Y,
    )

# Helper function for doing HMC inference
def run_inference(model, rng_key, X, Y, hypers, num_warmup, num_samples, num_chains):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y, hypers)
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()

# %% [markdown]
# # Prepare Data

# %%
def is_relevant_chromosome(chromosome):
    # Try running on all chromosomes, but uncomment other line if runtime is too long.
    return True
    # return chromosome in ["1", "14", "17", "21"]

donor_braak_dict = {}
with open("data/donor-info.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        donor_braak_dict[row["donor_id"]] = int(row["braak"])

rnaseq_donor_dict = {}
with open("data/columns-samples.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rnaseq_donor_dict[row["rnaseq_profile_id"]] = row["donor_id"]

gene_chromosome_dict = {}
gene_symbol_dict = {}
with open("data/rows-genes.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gene_chromosome_dict[row["gene_id"]] = row["chromosome"]
        gene_symbol_dict[row["gene_id"]] = row["gene_symbol"]

with open("data/fpkm_table_normalized.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rnaseq_profiles = header[1:]
    donors = [rnaseq_donor_dict[profile] for profile in rnaseq_profiles]
    braak_scores = [donor_braak_dict[donor] for donor in donors]
    Y = np.array(braak_scores)

    X = []
    gene_symbols = []
    for row in reader:
        gene_id = row[0]
        chromosome = gene_chromosome_dict[gene_id]
        if is_relevant_chromosome(chromosome):
            gene_symbols.append(gene_symbol_dict[gene_id])
            fpkms = [float(val) for val in row[1:]]
            X.append(fpkms)
    gene_symbols = np.array(gene_symbols)
    X = np.array(X).T

print("Shape of X: ", X.shape)
print("Shape of Y: ", Y.shape)

# %%
var_thresh = 0.05
num_select = 100

variances = np.var(X, axis=0)
print(f"# Genes with Variance > {var_thresh}: ", np.sum(variances > var_thresh))

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(variances, bins=np.linspace(0, 1, 100))
ax.axvline(var_thresh, color="red", linestyle="--")
ax.set_xlabel("Variance")
ax.set_ylabel("Frequency")
plt.show()

# %%
variance_selector = VarianceThreshold(var_thresh)
X = variance_selector.fit_transform(X)
selected_indices = variance_selector.get_support(indices=True)
gene_symbols = gene_symbols[selected_indices]

# TODO: Should we be using a different feature selection method?
select_k_best = SelectKBest(k=num_select)
X = select_k_best.fit_transform(X, Y)
selected_indices = select_k_best.get_support(indices=True)
gene_symbols = gene_symbols[selected_indices]

print("Shape of X: ", X.shape)

# %% [markdown]
# # Run Model

# %%
num_data = X.shape[0]
num_dimensions = X.shape[1]
num_active = 10
num_samples = 1000
num_warmup = 500
device = xla_bridge.get_backend().platform
num_chains = jax.device_count() if device == "gpu" else 1

print(f"Running {num_chains} chains on {device}")
numpyro.set_host_device_count(num_chains)
numpyro.set_platform(device)

hypers = {
    "expected_sparsity": num_active,
    "alpha1": 3.0,
    "beta1": 1.0,
    "alpha2": 3.0,
    "beta2": 1.0,
    "alpha3": 1.0,
    "c": 1.0,
}

rng_key = random.PRNGKey(0)
samples = run_inference(
    model, rng_key, X, Y, hypers, num_warmup, num_samples, num_chains
)

# %%
date = datetime.datetime.now().strftime("%m%d-%H%M%S")
with open(f"out/mcmc-{date}.pkl", "wb") as f:
    pickle.dump(samples, f)
with open(f"out/genes-{date}.pkl", "wb") as f:
    pickle.dump(gene_symbols, f)

# %% [markdown]
# # Interpret Results

# %%
# Compute the mean and variance of coefficient theta_i (where i = dimension) for a
# MCMC sample of the kernel hyperparameters (eta1, xisq, ...).
# Compare to theorem 5.1 in reference [1].
def compute_singleton_mean_variance(X, Y, dimension, msq, lam, eta1, xisq, c, sigma):
    P, N = X.shape[1], X.shape[0]

    probe = jnp.zeros((2, P))
    probe = probe.at[:, dimension].set(jnp.array([1.0, -1.0]))

    eta2 = jnp.square(eta1) * jnp.sqrt(xisq) / msq
    kappa = jnp.sqrt(msq) * lam / jnp.sqrt(msq + jnp.square(eta1 * lam))

    kX = kappa * X
    kprobe = kappa * probe

    k_xx = kernel(kX, kX, eta1, eta2, c) + sigma**2 * jnp.eye(N)
    k_xx_inv = jnp.linalg.inv(k_xx)
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)

    vec = jnp.array([0.50, -0.50])
    mu = jnp.matmul(k_probeX, jnp.matmul(k_xx_inv, Y))
    mu = jnp.dot(mu, vec)

    var = k_prbprb - jnp.matmul(k_probeX, jnp.matmul(k_xx_inv, jnp.transpose(k_probeX)))
    var = jnp.matmul(var, vec)
    var = jnp.dot(var, vec)

    return mu, var


# Compute the mean and variance of coefficient theta_ij for a MCMC sample of the
# kernel hyperparameters (eta1, xisq, ...). Compare to theorem 5.1 in reference [1].
def compute_pairwise_mean_variance(X, Y, dim1, dim2, msq, lam, eta1, xisq, c, sigma):
    P, N = X.shape[1], X.shape[0]

    probe = jnp.zeros((4, P))
    probe = probe.at[:, dim1].set(jnp.array([1.0, 1.0, -1.0, -1.0]))
    probe = probe.at[:, dim2].set(jnp.array([1.0, -1.0, 1.0, -1.0]))

    eta2 = jnp.square(eta1) * jnp.sqrt(xisq) / msq
    kappa = jnp.sqrt(msq) * lam / jnp.sqrt(msq + jnp.square(eta1 * lam))

    kX = kappa * X
    kprobe = kappa * probe

    k_xx = kernel(kX, kX, eta1, eta2, c) + sigma**2 * jnp.eye(N)
    k_xx_inv = jnp.linalg.inv(k_xx)
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)

    vec = jnp.array([0.25, -0.25, -0.25, 0.25])
    mu = jnp.matmul(k_probeX, jnp.matmul(k_xx_inv, Y))
    mu = jnp.dot(mu, vec)

    var = k_prbprb - jnp.matmul(k_probeX, jnp.matmul(k_xx_inv, jnp.transpose(k_probeX)))
    var = jnp.matmul(var, vec)
    var = jnp.dot(var, vec)

    return mu, var


# Sample coefficients theta from the posterior for a given MCMC sample.
# The first P returned values are {theta_1, theta_2, ...., theta_P}, while
# the remaining values are {theta_ij} for i,j in the list `active_dims`,
# sorted so that i < j.
def sample_theta_space(X, Y, active_dims, msq, lam, eta1, xisq, c, sigma):
    P, N, M = X.shape[1], X.shape[0], len(active_dims)
    # the total number of coefficients we return
    num_coefficients = P + M * (M - 1) // 2

    probe = jnp.zeros((2 * P + 2 * M * (M - 1), P))
    vec = jnp.zeros((num_coefficients, 2 * P + 2 * M * (M - 1)))
    start1 = 0
    start2 = 0

    for dim in range(P):
        probe = probe.at[start1 : start1 + 2, dim].set(jnp.array([1.0, -1.0]))
        vec = vec.at[start2, start1 : start1 + 2].set(jnp.array([0.5, -0.5]))
        start1 += 2
        start2 += 1

    for dim1 in active_dims:
        for dim2 in active_dims:
            if dim1 >= dim2:
                continue
            probe = probe.at[start1 : start1 + 4, dim1].set(
                jnp.array([1.0, 1.0, -1.0, -1.0])
            )
            probe = probe.at[start1 : start1 + 4, dim2].set(
                jnp.array([1.0, -1.0, 1.0, -1.0])
            )
            vec = vec.at[start2, start1 : start1 + 4].set(
                jnp.array([0.25, -0.25, -0.25, 0.25])
            )
            start1 += 4
            start2 += 1

    eta2 = jnp.square(eta1) * jnp.sqrt(xisq) / msq
    kappa = jnp.sqrt(msq) * lam / jnp.sqrt(msq + jnp.square(eta1 * lam))

    kX = kappa * X
    kprobe = kappa * probe

    k_xx = kernel(kX, kX, eta1, eta2, c) + sigma**2 * jnp.eye(N)
    L = cho_factor(k_xx, lower=True)[0]
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)

    mu = jnp.matmul(k_probeX, cho_solve((L, True), Y))
    mu = jnp.sum(mu * vec, axis=-1)

    Linv_k_probeX = solve_triangular(L, jnp.transpose(k_probeX), lower=True)
    covar = k_prbprb - jnp.matmul(jnp.transpose(Linv_k_probeX), Linv_k_probeX)
    covar = jnp.matmul(vec, jnp.matmul(covar, jnp.transpose(vec)))

    # sample from N(mu, covar)
    L = jnp.linalg.cholesky(covar)
    sample = mu + jnp.matmul(L, np.random.randn(num_coefficients))

    return sample


# Get the mean and variance of a gaussian mixture
def gaussian_mixture_stats(mus, variances):
    mean_mu = jnp.mean(mus)
    mean_var = jnp.mean(variances) + jnp.mean(jnp.square(mus)) - jnp.square(mean_mu)
    return mean_mu, mean_var


# Helper function for analyzing the posterior statistics for coefficient theta_i
def analyze_dimension(samples, X, Y, dimension, hypers):
    vmap_args = (
        samples["msq"],
        samples["lambda"],
        samples["eta1"],
        samples["xisq"],
        samples["sigma"],
    )
    mus, variances = vmap(
        lambda msq, lam, eta1, xisq, sigma: compute_singleton_mean_variance(
            X, Y, dimension, msq, lam, eta1, xisq, hypers["c"], sigma
        )
    )(*vmap_args)
    mean, variance = gaussian_mixture_stats(mus, variances)
    std = jnp.sqrt(variance)
    return mean, std


# Helper function for analyzing the posterior statistics for coefficient theta_ij
def analyze_pair_of_dimensions(samples, X, Y, dim1, dim2, hypers):
    vmap_args = (
        samples["msq"],
        samples["lambda"],
        samples["eta1"],
        samples["xisq"],
        samples["sigma"],
    )
    mus, variances = vmap(
        lambda msq, lam, eta1, xisq, sigma: compute_pairwise_mean_variance(
            X, Y, dim1, dim2, msq, lam, eta1, xisq, hypers["c"], sigma
        )
    )(*vmap_args)
    mean, variance = gaussian_mixture_stats(mus, variances)
    std = jnp.sqrt(variance)
    return mean, std

# %%
# compute the mean and square root variance of each coefficient theta_i
dims = jnp.arange(num_dimensions)
means, stds = vmap(lambda dim: analyze_dimension(samples, X, Y, dim, hypers))(dims)

# %%
active_dimensions = []
for dim, (mean, std) in enumerate(zip(means, stds)):
    # mark dimension inactive if interval [mean +/- 3 * std] contains zero
    lower, upper = mean - 3.0 * std, mean + 3.0 * std
    if lower * upper > 0.0:
        active_dimensions.append(dim)
        print(f"theta[{dim + 1}]: {mean:.2e} +- {std:.2e}")

print(f"Identified a total of {len(active_dimensions)} active dimensions")

# Compute the mean and square root variance of coefficients theta_ij for i,j active dimensions.
# Note that the resulting numbers are only meaningful for i != j.
if len(active_dimensions) > 0:
    dim_pairs = jnp.array(list(itertools.product(active_dimensions, active_dimensions)))
    means, stds = vmap(
        lambda dim_pair: analyze_pair_of_dimensions(
            samples, X, Y, dim_pair[0], dim_pair[1], hypers
        )
    )(dim_pairs)
    for dim_pair, mean, std in zip(dim_pairs, means, stds):
        dim1, dim2 = dim_pair
        if dim1 >= dim2:
            continue
        lower, upper = mean - 3.0 * std, mean + 3.0 * std
        if lower * upper > 0.0:
            print(f"theta[{dim1 + 1}, {dim2 + 1}]: {mean:.2e} +- {std:.2e}]")

    # Draw a single sample of coefficients theta from the posterior, where we return all singleton
    # coefficients theta_i and pairwise coefficients theta_ij for i, j active dimensions. We use the
    # final MCMC sample obtained from the HMC sampler.
    thetas = sample_theta_space(
        X,
        Y,
        active_dimensions,
        samples["msq"][-1],
        samples["lambda"][-1],
        samples["eta1"][-1],
        samples["xisq"][-1],
        hypers["c"],
        samples["sigma"][-1],
    )
    print("Single posterior sample theta:\n", thetas)

# %%



