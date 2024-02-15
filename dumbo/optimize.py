from botorch.acquisition import AnalyticAcquisitionFunction
import numpy as np
import multiprocessing as mp
import torch
from botorch.optim import optimize_acqf, gen_batch_initial_conditions
import gpytorch
from dumbo.decomposition_sampling import *
import warnings
import multiprocessing

class DumboAcqFunction(AnalyticAcquisitionFunction):
	r"""
	The acquisition function used by a DuMBO optimizer object to find the next query. It is discussed in [1] (Sections 4
	and 5).

	Inherit from botorch.acquisition.AnalyticAcquisitionFunction.
	"""
	def __init__(self, model, d, n_neighbors, n_models, shared_var, t, xm, lambdas, eta):
		"""Initialize the DuMBO acquisition function.

		Args:
				model (GPFactor): a factor in one of the inferred additive decompositions
				d (int): the number of dimensions of the objective function
				n_neighbors (int): the number of factors that share their variances with `model`
				n_models (int): the number of inferred additive decompositions
				shared_var (float): the sum of the variances shared with `model`
				t (int): the time index
				xm (torch.tensor): the consensus variable (see [2] Eq. 11), of shape `(d_i,)` if the factor is
				`d_i`-dimensional. If None, the acquisition function does not compute the Lagrangian penalty
				lambdas (array): the dual variables (see [2], Eq. 11) of shape `(d_i,)` if the factor is `d_i`-dimensional. If
				None, the acquisition function does not compute the Lagrangian penalty
				eta (float): the weight of the quadratic penalty (see [2], Eq. 11). If None, the acquisition function does not
				compute the Lagrangian penalty
		"""
		super(AnalyticAcquisitionFunction, self).__init__(model)
		self.beta = np.sqrt(0.2 * d * np.log(2 * t))
		self.xm = xm if xm is not None else xm
		self.lambdas = torch.reshape(lambdas, (-1, 1)) if lambdas is not None else lambdas
		self.eta = eta
		self.n_neighbors = n_neighbors
		self.n_models = n_models
		self.shared_var = shared_var

	def local_constraints(self, x):
		"""Evaluate the Lagrangian penalty for an input `x`.

		Args:
				x (array): the input, of shape `(d_i,)` if the current factor is `d_i`-dimensional

		Returns:
				float: the Lagrangian penalty at input `x`.
		"""
		if self.xm is None:
			return None
		x_xm = x - self.xm
		return torch.reshape(torch.matmul(x_xm, self.lambdas), (-1,)) + self.eta * torch.sum(x_xm * x_xm, dim=-1) / 2.0

	def forward(self, x):
		"""Compute the acquisition function value for an input `x`.

		Args:
				x (array): the input, of shape `(d_i,)`, `(n_inputs,d_i)` or `(n_inputs,n_batches,d_i)` with `n_inputs` the
				number of inputs, `n_batches` the number of batches ands `d_i` the dimensionality of the current factor

		Returns:
				array or float: the acquisition function values. Float if a single input is passed, array of shape (n_inputs,)
				otherwise
		"""
		if len(x.shape) == 3:
			x = torch.reshape(x, (x.shape[0], x.shape[2]))
		
		c = self.local_constraints(x)
		mean, covar = self.model.mean(x), self.model.posterior_var(x)

		# Acquisition function
		dumbo_ucb = mean + self.beta * torch.sqrt(covar / (self.n_neighbors ** 2) + self.shared_var)

		# Lagrangian penalty
		if c is not None:
			dumbo_ucb -= c
		
		final_res = dumbo_ucb / self.n_models
		return final_res


class MyOwnGP(gpytorch.models.ExactGP):
	r"""
	The homemade additive GP model serving as a surrogate model for the objective function.

	Inherit from gpytorch.models.ExactGP for kernels' hyperparameters inference.
	"""
	num_outputs = 1

	def __init__(self, kernel, train_x, train_y, likelihood):
		"""Build the additive GP model

		Args:
				kernel (gpytorch.kernels.Kernel): the additive kernel
				train_x (torch.Tensor): the training observations in the input space
				train_y (torch.Tensor): the training observations in the output space
				likelihood (gpytorch.mlls.Likelihood): the likelihood used for hyperparameters inference
		"""
		super(MyOwnGP, self).__init__(train_x, train_y, likelihood)
		self.train_x = train_x
		self.train_y = train_y
		self.likelihood = likelihood
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = kernel

	def forward(self, x):
		"""Build the multivariate gaussian distribution underpinning the Gaussian process.

		Args:
				x (torch.Tensor): the input, of shape `(d,)` or `(n_inputs,d)` with `n_inputs` the number of inputs and `d` the
				dimensionality of the objective function

		Returns:
				gpytorch.distributions.MultivariateNormal: the distribution on the provided input `x`.
		"""
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
	
	def fit(self):
		"""Hyperparameters inference of the model.
		"""
		# Find optimal model hyperparameters
		self.train()
		self.likelihood.train()

		# Use the adam optimizer
		optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

		# "Loss" for GPs - the marginal log likelihood
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

		n_iterations = 200
		for i in range(n_iterations):
			# Zero gradients from previous iteration
			optimizer.zero_grad()
			# Output from model
			output = self(self.train_x)
			# Calc loss and backprop gradients
			loss = -mll(output, self.train_y)
			loss.backward()
			optimizer.step()

		self.eval()
		self.likelihood.eval()
	
	def posterior(self, X, posterior_transform=None):
		return self.likelihood(self(X.double()))
		
	def get_mll(self):
		"""Compute the marginal log likelihood of the model.

		Returns:
				float: the marginal log likelihood of the model.
		"""
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
		output = self(self.train_x)
		return mll(output, self.train_y).detach().numpy()
	
	def extract_kernels(self):
		"""Return the list of kernels involved in the additive model.

		Returns:
				list: list of gpytorch.kernels.Kernel
		"""
		if hasattr(self.covar_module, "kernels"):
			return self.covar_module.kernels
		return [self.covar_module]

	def gram_inverse(self):
		"""Compute the inverse of the Gram matrix on the training observations.

		Returns:
				torch.Tensor: the inverse of the Gram matrix, of shape `(n,n)` for `n` observations
		"""
		return torch.inverse(self.covar_module(self.train_x).evaluate())
	
	@staticmethod
	def model_from_decomposition(decomposition, X, y, base_kernel_class, base_kernel_args):
		"""From the provided additive decomposition `decomposition`, build an additive gaussian process

		Args:
				decomposition (list): a list of one np.array per factor of the decomposition, containing the dimension indices
				of the factor
				X (torch.Tensor): the training observations in the input space, shape `(n,d)` for `n` observations and
				`d`-dimensional input space
				y (torch.Tensor): the training observations in the output space, shape `(n,)` for `n` observations
				base_kernel_class (class from gpytorch.kernels.Kernel): the base kernel in the additive model.
				base_kernel_args (list): the arguments for the base kernel.

		Returns:
				MyOwnGP: An additive gaussian process based on the provided additive decomposition
		"""
		kernel = None
		for factor in decomposition:
			new_k = gpytorch.kernels.ScaleKernel(base_kernel_class(*base_kernel_args, active_dims=torch.tensor(factor, dtype=torch.int32)))
			if kernel is None:
				kernel = new_k
			else:
				kernel += new_k

		model = MyOwnGP(kernel, X, y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
		return model
	

class MyGPFactor(gpytorch.models.ExactGP):
	r"""
	The surrogate model of a factor in the additive decomposition.

	Inherit from ExactGP for compatibility.
	"""
	def __init__(self, intervals, kernel, train_x, train_y, invK, t, neighbors, n_models, likelihood):
		"""Build the GP factor.

		Args:
				intervals (torch.Tensor): list of infimum and supremum, of shape `(d,2)` if the objective function is
				`d`-dimensional
				kernel (gpytorch.kernels.Kernel): the kernel used by the factor with the hyperparameters inference already
				performed
				train_x (torch.Tensor): the training observations in the input space, of shape `(n,d)` for `n` observations
				train_y (torch.Tensor): the training observations in the output space, of shape `(n,)` for `n` observations
				invK (torch.Tensor): the inverse of the Gram matrix generated by the whole additive model, of shape `(n,n)` for
				`n` observations
				t (int): the time index
				neighbors (list): the list of factors sharing their variances with the current factor
				n_models (int): the number of additive decompositions inferred
				likelihood (gpytorch.likelihoods.Likelihood): the likelihood for compatibility with the parent class
		"""
		self.d = len(train_x[0])
		train_x = train_x[:, kernel.active_dims.type(torch.long)]
		train_y = train_y
		super(MyGPFactor, self).__init__(train_x, train_y, likelihood)

		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = kernel

		self.train_x = train_x
		self.padded_train_x = self.x_padding(train_x)
		self.train_y = train_y
		self.intervals = intervals

		self.invK = invK
		self.shared_variance = 0.0
		self.neighbors = neighbors
		self.n_neighbors = len(neighbors)
		self.n_models = n_models
		self.t = t
		self.eval()

	def x_padding(self, X):
		"""Padding for a matrix of shape `(n,d_i)` comprising `n` `d_i`-dimensional inputs for the factor into a matrix of shape
		`(n,d)` with 0 in their inactive dimensions.

		Args:
				X (torch.Tensor): the inputs requiring padding, of shape `(n,d_i)`

		Returns:
				torch.Tensor: padded inputs of shape `(n,d)`
		"""
		if len(X.shape) == 1:
			X = torch.reshape(X, (1, -1))
		padded = torch.zeros((X.shape[0], self.d,), dtype=torch.double)
		padded[:, self.active_dims()] = X

		return padded

	def mean(self, X):
		"""Posterior mean of the factor.

		Args:
				X (torch.Tensor): inputs for the posterior mean, with last dimension of size `d_i`

		Returns:
				torch.Tensor: the posterior mean at the provided inputs
		"""
		padded_X = self.x_padding(X)
		kX = self.covar_module(padded_X, self.padded_train_x).evaluate()
		return torch.reshape(torch.matmul(torch.matmul(kX, self.invK), self.train_y), (-1,))

	def posterior_var(self, X):
		"""Posterior variance of the factor.

		Args:
				X (torch.Tensor): inputs for the posterior variance, with last dimension of size `d_i`

		Returns:
				torch.Tensor: the posterior variance at the provided inputs
		"""
		padded_X = self.x_padding(X)
		kX = self.covar_module(padded_X, self.padded_train_x).evaluate()
		kXX = self.covar_module(torch.reshape(padded_X, (1,-1))).evaluate()[0, 0] * torch.ones(padded_X.size()[0])

		jitter = 1e-5
		final_result = kXX - torch.reshape(torch.sum(torch.mul(torch.matmul(kX, self.invK), kX), dim=1), (-1,))
		return torch.maximum(final_result, jitter * torch.ones(final_result.shape))
	
	def covar(self, X):
		"""Covariance function of the factor.

		Args:
				X (torch.Tensor): inputs for the covariance function, with last dimension of size `d_i`

		Returns:
				torch.Tensor: the covariance matrix at the provided inputs
		"""
		padded_X = self.x_padding(X)
		return self.covar_module(padded_X).evaluate()
	
	def share_variance(self, variance):
		"""Set the shared variance to a new value

		Args:
				variance (float): the sum of the shared variance
		"""
		self.shared_variance = variance

	def dumbo_acq(self, xm, lambdas, eta):
		"""Return the acquisition function of this factor.

		Args:
				xm (torch.Tensor): the consensus variable of shape `(d_i,)`
				lambdas (torch.Tensor): the dual variables of shape `(d_i,)`
				eta (float): the weight for the quadratic penalty

		Returns:
				DumboAcqFunction: the acquisition function for this factor
		"""
		return DumboAcqFunction(self, self.d, self.n_neighbors, self.n_models, self.shared_variance, self.t, xm, lambdas, eta)
	
	def next_candidate(self, xm, lambdas, eta, x=None):
		"""Maximize the acquisition function given the Lagrangian variable and the previously found candidate if provided.

		Args:
				xm (torch.Tensor): the consensus variable of shape `(d_i,)`
				lambdas (torch.Tensor): the dual variables of shape `(d_i,)`
				eta (float): the weight for the quadratic penalty
				x (torch.Tensor, optional): the previously found candidate. Defaults to None.

		Returns:
				torch.Tensor: the maximal argument of the acquisition function, of shape `(d_i,)`
		"""
		# Acquisition function and bounds
		acqf = self.dumbo_acq(xm, lambdas, eta)
		bounds = self.bounds()

		# If the previously found candidate is not provided, multi-start optimization, otherwise single start
		n_restarts = 20 if x is None else 1
		candidates = gen_batch_initial_conditions(acqf, bounds=bounds, q=1, num_restarts=n_restarts, raw_samples=512)
		if x is not None:
			# The previously found candidate is necessarily a starting point
			candidates = torch.cat((torch.reshape(x, (1,1,-1)), candidates))

		candidate, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=20, raw_samples=512, batch_initial_conditions=candidates)
		return candidate[0]
	
	def dependency_vector(self):
		"""Show the active dimensions of the factor as a vector of shape `(d,)` comprising ones for active dimensions and
		zeros for inactive dimensions.

		Returns:
				np.array: the active dimensions flagged as ones in a `(d,)` vector
		"""
		dep_vec = np.zeros((self.d,), dtype=np.int16)
		dep_vec[self.active_dims()] = 1

		return dep_vec
	
	def active_dims(self):
		"""Return the indices of the active dimensions of the factor

		Returns:
				np.array: the indices of the active dimensions of the factor
		"""
		return self.covar_module.active_dims.detach().numpy()
	
	def bounds(self):
		"""Return the bounds of the active parameters for the factor

		Returns:
				torch.Tensor: the bounds of the active parameters
		"""
		ad = self.active_dims()
		lower_bounds = torch.tensor([self.intervals[idx][0] for idx in ad], dtype=torch.double)
		upper_bounds = torch.tensor([self.intervals[idx][1] for idx in ad], dtype=torch.double)

		return torch.stack([lower_bounds, upper_bounds])

	def consensus_diff(self, x, xm=None):
		"""Compute the difference between the input and the consensus variable if provided. Return None otherwise.

		Args:
				x (torch.Tensor): the input
				xm (torch.Tensor, optional): the consensus variable. Must be of the same shape as `x`. Defaults to None.

		Returns:
				torch.Tensor: the elementwise difference between `x` and `xm` if `xm` is not None. None otherwise.
		"""
		return x - xm if xm is not None else None
	
	def acq(self, x, xm, lambdas, eta):
		"""Compute the acquisition function of the factor at the provided input `x`.

		Args:
				x (torch.Tensor): the input, of shape `(d_i,)`
				xm (torch.Tensor): the consensus variable, of shape `(d_i,)`
				lambdas (torch.Tensor): the dual variables, of shape `(d_i,)
				eta (float): the weight for the quadratic penalty

		Returns:
				float: the acquisition function value at input x
		"""
		return self.dumbo_acq(xm, lambdas, eta).forward(x)


class ADMM:
	r"""
	The ADMM class, useful to find candidates for each factor of the additive decomposition and aggregate them into one
	final query recommendation.
	"""

	def __init__(self, factors, abstol=0.05, max_it=10, initial_eta=1.0, n_cores=None):
		"""Build the ADMM object

		Args:
				factors (list): the list of GPFactors
				abstol (float, optional): the aggregation precision as a stopping criterion. Defaults to 0.05.
				max_it (int, optional): the maximal number of ADMM iteration as stopping criterion. Defaults to 10.
				initial_eta (float, optional): the initial weight for the quadratic penalty. Defaults to 1.0.
				n_cores (int, optional): the number of cores available for parallel computation. If `None` is provided, it is
				set to the return value of `multiprocessing.cpu_count()`. Defaults to None.
		"""
		self._factors = factors
		self._abstol = abstol
		self._max_it = max_it
		self._initial_eta = initial_eta

		self._factor_variable_matrix = self.build_factor_variable_matrix()
		self._n = self._factor_variable_matrix.shape[0]
		self._d = self._factor_variable_matrix.shape[1]
		
		self._n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()

	def build_factor_variable_matrix(self):
		"""Build the factor-variable matrix, also known as the adjacency matrix of the factor graph.

		Returns:
				np.array: the factor-variable matrix, of shape `(n,d)` for `n` factors and a `d`-dimensional objective function
		"""
		matrix = []
		for gp in self._factors:
			matrix.append(gp.dependency_vector())

		return np.array(matrix)
	
	def consensus_differences(self, x, xm):
		"""Evaluate the difference between the input and the consensus variable if provided for each factor.

		Args:
				x (list): list of tensors as inputs for factor consensus difference
				xm (torch.Tensor): the consensus variable of shape (d,)

		Returns:
				list: list of tensors comprising the elementwise difference between the input and the consensus variable for
				each factor
		"""
		constraint_evaluation = []
		for i,gp in enumerate(self._factors):
			constraint_evaluation.append(gp.consensus_diff(x[i], xm[gp.active_dims()]))
		return constraint_evaluation
	
	def residuals(self, consensus_diff, xm_prev, xm, eta):
		"""Compute the residuals for ADMM optimization.

		Args:
				consensus_diff (list): list of elementwise difference with the consensus variable for each factor
				xm_prev (torch.Tensor): the previously found consensus, of shape `(d,)`
				xm (torch.Tensor): the current consensus, of shape `(d,)`
				eta (float): the weight for the quadratic penalty

		Returns:
				tuple: couple of residuals
		"""
		r = np.sqrt(sum([np.linalg.norm(consensus_diff_i) ** 2 for consensus_diff_i in consensus_diff]))
		s = np.linalg.norm(eta * (xm - xm_prev))
		return r, s
	
	def update_quadratic_penalty(self, eta, r, s, f=10.0, tau=2.0):
		"""Update the weight for the quadratic penalty according to the residuals.

		Args:
				eta (float): the current weight for the quadratic penalty
				r (float): the first residual
				s (float): the second residual
				f (float, optional): the value that the ratio of the residuals mustn't exceed. Defaults to 10.0.
				tau (float, optional): the multiplicative factor for the update of the penalty. Defaults to 2.0.

		Returns:
				float: the updated weight for the quadratic penalty
		"""
		if r > f * s:
			return eta * tau
		if s > f * r:
			return eta / tau

		return eta
	
	@staticmethod
	def parallel_optimization(gp, xm, lambdas, eta, x):
		"""Find the next candidate for a given factor of the additive decomposition

		Args:
				gp (GPFactor): the surrogate model for a factor of the additive decomposition
				xm (torch.Tensor): the consensus variable, of shape `(d_i,)` if the factor is `d_i`-dimensional
				lambdas (torch.Tensor): the dual variables, of shape `(d_i,)` if the factor is `d_i`-dimensional
				eta (float): the weight for the quadratic penalty
				x (torch.Tensor): the input for the given factor, of shape `(d_i)` if the factor is `d_i`-dimensional

		Returns:
				torch.tensor: the next candidate for the provided factor
		"""
		return gp.next_candidate(xm, lambdas, eta, x)

	def aggregate_candidates(self, x):
		"""Gather the recommendations made by the factors for each dimension

		Args:
				x (list): the list of the candidates for each factor

		Returns:
				list: list of list of recommendations for each single dimension
		"""
		data = [[] for _ in range(self._d)]
		for i,xi in enumerate(x):
			for j,dij in enumerate(self._factors[i].active_dims()):
				data[dij].append(xi[j])

		return [torch.tensor(d, dtype=torch.double) for d in data]

	def update_xm(self, x):
		"""Aggregate the candidates for each factor into a single, averaged recommendation

		Args:
				x (list): the list of candidates for each factor

		Returns:
				torch.tensor: the averaged recommendation
		"""
		data = self.aggregate_candidates(x)
		for i,di in enumerate(data):
			data[i] = torch.sum(di) / di.size()[0]

		return torch.tensor(data, dtype=torch.double)

	def admm_acquisition(self, x, xm, lambdas, eta):
		"""Compute the sum of each factor's acquisition function.

		Args:
				x (list): the list of inputs, one for each factor
				xm (torch.Tensor): the consensus variable, of shape `(d,)`
				lambdas (list): list of dual variables for each factor
				eta (float): the weight for the quadratic penalty

		Returns:
				float: the sum of each factor's acquisition function
		"""
		f = 0
		for i,gp in enumerate(self._factors):
			f += gp.acq(x[i], xm[gp.active_dims()], lambdas[i], eta)

		return f

	def find_candidates(self, xm, lambdas, eta, x):
		"""Find the candidates for each factor in parallel.

		Args:
				xm (torch.Tensor): the consensus variable, of shape `(d,)`
				lambdas (list): list of dual variables for each factor
				eta (float): the weight for the quadratic penalty
				x (list): the list of previously found candidates for each factor

		Returns:
				list: the list of candidates for each factor
		"""
		all_candidates = None
		with mp.Pool(processes=self._n_cores) as processes:
			args = [(gp, xm[gp.active_dims()] if xm is not None else None, lambdas[i], eta, x[i]) for i,gp in enumerate(self._factors)]
			all_candidates = processes.starmap(ADMM.parallel_optimization, args)

		return all_candidates

	def next_query(self):
		"""Find the next query according to ADMM (see [4])

		Returns:
				torch.tensor: the next query for the black-box objective function
		"""
		n_external_iterations = 0

		all_constraints_satisfied = False
		constraints_evaluated = []

		# Consensus variable and previously found candidates
		xm = None
		x = [None for _ in range(self._n)]

		# Dual variables
		lambdas = []
		for gp in self._factors:
			lambdas.append(torch.zeros(len(gp.active_dims()), dtype=torch.double))

		# Quadratic penalty
		eta = self._initial_eta

		# ADMM iteration
		while not all_constraints_satisfied:
			n_external_iterations += 1

			x = self.find_candidates(xm, lambdas, eta, x)
			with torch.no_grad():
				# Once the candidates have been found, share their variances
				variances = np.array([gp.posterior_var(torch.reshape(x[i], (1,-1)))[0] for i,gp in enumerate(self._factors)])
				gps_n_neighbors = np.array([gp.n_neighbors ** 2 for gp in self._factors])
				for i,gp in enumerate(self._factors):
					if len(gp.neighbors) > 1:
						gi_minus_i = gp.neighbors.difference({i})
						gp.share_variance(np.sum(variances[np.sort(np.array(list(gi_minus_i)))] / gps_n_neighbors[np.sort(np.array(list(gi_minus_i)))]))
					else:
						gp.share_variance(0.0)
				
				# The unconstrained problem is solved
				# Update the consensus and dual variables
				xm_prev = xm
				xm = self.update_xm(x)

				constraints_evaluated = self.consensus_differences(x, xm)
				# Lambdas update
				for i in range(self._n):
					lambdas[i] += eta * constraints_evaluated[i]

				# Eta update
				if xm_prev is not None:
					r, s = self.residuals(constraints_evaluated, xm_prev, xm, eta)
					eta = self.update_quadratic_penalty(eta, r, s)

				infty_norm_constraints = max([torch.max(torch.abs(ci)) for ci in constraints_evaluated])
				all_constraints_satisfied = infty_norm_constraints < self._abstol or n_external_iterations >= self._max_it

		return xm
	

class DuMBOOptimizer:
	r"""
	This class contains the DuMBO optimizer as described in [1] and is the main interface with the final user.
	"""

	def __init__(self, intervals, X=None, y=None, n_init_points=2, dmax=None, n_samples_per_iteration=5, precision=0.05, max_it=10, base_kernel_class=gpytorch.kernels.MaternKernel, base_kernel_args=[2.5], n_cores=None):
		"""Build the DuMBO optimizer

		Args:
				intervals (np.array): the array of infimum and supremum for each dimension, of shape `(d,2)`
				X (np.array, optional): the training observations in the input space, of shape `(n,d)` for `n` observations
				and `d`-dimensional inputs. Defaults to None.
				y (np.array, optional): the training observations in the output space, of shape `(n,)` for `n` observations.
				Defaults to None.
				n_init_points (int, optional): the number of points sampled randomly before starting to use ADMM. Cannot be
				lower than 2. Defaults to 2.
				dmax (int, optional): the maximal factor size for a factor of the inferred additive decomposition.
				Defaults to None.
				n_samples_per_iteration (int, optional): number of additive decomposition sampled at each iteration. Defaults to
				5.
				precision (float, optional): tolerance on the diversity of the candidates found by ADMM. Stopping criterion for
				ADMM. Defaults to 0.05.
				max_it (int, optional): number of maximal iterations allowed for ADMM. Stopping criterion for ADMM. Defaults to
				10.
				base_kernel_class (class from gpytorch.kernels.Kernel, optional): kernel used for each factor of the inferred
				additive decompositions. Defaults to gpytorch.kernels.MaternKernel.
				base_kernel_args (list, optional): arguments for the base kernel. Defaults to [2.5].
				n_cores (int, optional): the number of cores for parallel computation. If `None` is provided, it is set to the
				return value of `multiprocessing.cpu_count()`.
		"""
		intervals = torch.tensor(intervals, dtype=torch.double)
		self._normed_intervals = torch.zeros(intervals.size(), dtype=torch.double)
		self._normed_intervals[:, 1] = 1.0
		self._lower_bounds = intervals[:, 0]
		self._upper_bounds = intervals[:, 1]

		self._base_kernel_class = base_kernel_class
		self._base_kernel_args = base_kernel_args

		self._X = X if X is None else torch.tensor(X, dtype=torch.double)
		self._n_init_points = n_init_points - (0 if X is None else self._X.size()[0])
		self.normalize_X()

		self._y = y if y is None else torch.tensor(y, dtype=torch.double)
		self.standardize_y()

		self._precision = precision
		self._max_it = max_it

		self._d = len(intervals)
		self._dmax = self._d if dmax is None else dmax
		self.first_decomposition()

		self._t = 0 if self._X is None else self._X.shape[0]
		self._n_samples_per_iteration = n_samples_per_iteration if self._dmax > 1 else 1

		self._n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()

	def first_decomposition(self):
		if self._n_init_points <= 0:
			self._current_decomposition = [np.arange(self._d)] if self._d == self._dmax else random_additive_decomposition(self._d, self._dmax)
			self._current_model = MyOwnGP.model_from_decomposition(self._current_decomposition, self._norm_X, self._norm_y, base_kernel_class=self._base_kernel_class, base_kernel_args=self._base_kernel_args)
			self._current_model.fit()

	def standardize_y(self):
		"""Standardize the y vector.
		"""
		if self._y is not None and self._y.shape[0] > 1:
			self._norm_y = (self._y - torch.mean(self._y)) / torch.std(self._y)

	def normalize_X(self):
		"""Normalize the X matrix between 0 and 1
		"""
		if self._X is not None:
			self._norm_X = (self._X - self._lower_bounds) / (self._upper_bounds - self._lower_bounds)
	
	def denormalize_X(self, X):
		"""Denormalize the input to its original intervals.

		Args:
				X (torch.Tensor): normalized input with each element between 0 and 1

		Returns:
				torch.Tensor: the denormalized input
		"""
		return (self._upper_bounds - self._lower_bounds) * X + self._lower_bounds

	def random_normalized_configuration(self):
		"""Sample a normalized random input

		Returns:
				np.array: an input of shape `(d,)`
		"""
		return np.random.uniform(size=(self._d,))

	def next_query(self):
		"""Find the next query for the objective function.

		Returns:
				np.array: the next query
		"""
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			warnings.simplefilter("ignore", category=UserWarning)
			if self._n_init_points > 0:
				return self.denormalize_X(self.random_normalized_configuration()).numpy()
			
			if self._dmax > 1:
				# Sample n_samples_per_iteration additive decompositions
				decompositions, models = mcmc_sampling(self._current_decomposition, self._current_model, self._dmax, self._norm_X, self._norm_y, self._base_kernel_class, self._base_kernel_args, k=self._n_samples_per_iteration)
				self._current_decomposition = decompositions[-1]
				self._current_model = models[-1]
			else:
				decompositions, models = [self._current_decomposition], [self._current_model]

			factors = []

			# Build the neighboring data structure
			neighbors = []
			fis = []
			for i in range(self._d):
				fis.append(set(()))
				factor_shift = 0
				for dec in decompositions:
					for j,a in enumerate(dec):
						if i in a:
							fis[-1] = fis[-1].union({j + factor_shift})
					factor_shift += len(dec)
					
			for dec in decompositions:
				for factor in dec:
					neighbors.append(set(()))
					for d_idx in factor:
						neighbors[-1] = neighbors[-1].union(fis[d_idx])

			# Create the surrogate models for the factors of the inferred additive decompositions
			factor_shift = 0
			for dec,model in zip(decompositions, models):
				with torch.no_grad():
					kernels = model.extract_kernels()
					invK = model.gram_inverse()
				for i,k in enumerate(kernels):
					factors.append(MyGPFactor(self._normed_intervals, k, self._norm_X, self._norm_y, invK, self._t, neighbors[i+factor_shift], self._n_samples_per_iteration, gpytorch.likelihoods.GaussianLikelihood()))
				factor_shift += len(kernels)

			# Create ADMM and find the next query
			optimizer = ADMM(factors, abstol=self._precision, max_it=self._max_it, n_cores=self._n_cores)
			return self.denormalize_X(optimizer.next_query()).numpy()
	
	def tell(self, x, y):
		"""Append a new observation to the existing dataset, increase the time index and update the model for MCMC sampling.

		Args:
				x (np.array): the observation in the input space
				y (float): the observation in the output space
		"""
		self._t += 1
		self._X = torch.cat((self._X, torch.reshape(torch.tensor(x, dtype=torch.double), (1,-1)))) if self._X is not None else torch.reshape(torch.tensor(x, dtype=torch.double), (1,-1))

		# Check for duplicate input
		dup = False
		uniq_X = torch.unique(self._X, dim=0)
		if uniq_X.size(0) != self._X.size(0):
			# The input is a duplicate, don't add it to the dataset
			self._X = uniq_X
			dup = True

		self.normalize_X()
		if not dup:
			self._y = torch.cat((self._y, torch.tensor([y], dtype=torch.double))) if self._y is not None else torch.tensor([y], dtype=torch.double)
		self.standardize_y()

		if self._n_init_points > 0:
			self._n_init_points -= 1
			self.first_decomposition()
		else:
			current_ker_model = self._current_model.covar_module
			current_likelihood = self._current_model.likelihood
			self._current_model = MyOwnGP(current_ker_model, self._norm_X, self._norm_y, current_likelihood)
			self._current_model.fit()