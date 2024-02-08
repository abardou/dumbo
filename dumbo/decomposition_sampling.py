import numpy as np
import random
import dumbo.optimize
import gpytorch

r"""
Methods for sampling additive decomposition according to Gardner et. al. (2017) (see [3] in README.md at the root
directory)
"""

# Cache to save some computing time when counting partitions
__cache__decomposition = {}

def count_partitions(n, limit):
	"""Count the number of partitions of `n` elements with at most `limit` elements per subset. For more efficient
	response time, this function uses the global variable `__cache__decomposition`.

	Args:
			n (int): the number of elements
			limit (int): the maximal number of elements per subset

	Returns:
			int: the number of partitions of `n` elements with at most `limit` elements per subset.
	"""
	if n == 0:
			return 1
	if (n, limit) in __cache__decomposition:
			return __cache__decomposition[n, limit]
	
	x = __cache__decomposition[n, limit] = sum(count_partitions(n-k, k) for k in range(1, min(limit, n) + 1))
	return x

def random_partition(n):
	"""Sample a random partition of `n` elements.

	Args:
			n (int): the number of elements

	Returns:
			list: a random partition of `n` elements
	"""
	p = []
	limit = n
	total = count_partitions(n, limit)
	which = random.randrange(total)
	while n:
		for k in range(1, min(limit, n) + 1):
				count = count_partitions(n-k, k)
				if which < count:
						break
				which -= count
		p.append(k)
		limit = k
		n -= k
	return p

def random_additive_decomposition(d, dmax):
	"""Sample a random additive decomposition for a `d`-dimensional function. Each factor of the decomposition will be at
	most `dmax`-dimensional.

	Args:
			d (int): the number of dimensions
			dmax (int): the maximum factor size of the returned decomposition

	Returns:
			list: a list of one numpy.array per factor of the decomposition, containing the dimension indices of the factor
	"""
	dims = np.arange(d)
	np.random.shuffle(dims)
	saved = np.array([])
	while d > 0:
		rp = random_partition(d)

		# Identify where dmax lies
		exploitable = False
		for i,di in enumerate(rp):
			if di <= dmax:
				exploitable = True
				break
		if exploitable:
			d -= np.sum(rp[i:])
			saved = np.concatenate((rp[i:], saved), axis=None)

	return sort_dimensions(np.split(dims, np.cumsum(saved[:-1], dtype=int)))

def list_potential_merges(lens, dmax):
	"""List all the couples of factors that can be merged into one without being more than `dmax`-dimensional.

	Args:
			lens (list): dimensionality of the factors, list of integers with shape `(n,)` if the decomposition has `n`
			factors
			dmax (int): the maximal factor size of the decomposition

	Returns:
			list: list of couples of factor indices that can be merged
	"""
	couples = []
	for i,ldi in enumerate(lens):
		# The factor i can be merged with at least a 1-dimensional factor
		if ldi < dmax:
			for j,ldj in enumerate(lens[i+1:]):
				# The factors i and j can be merged without violating the constraint
				if ldj <= dmax - ldi:
					couples.append((i,i+1+j))
	return couples

def is_merge_allowed(lens, dmax):
	"""Test the existence of at least one couple of factors that can be merged into one without being more than
	`dmax`-dimensional

	Args:
			lens (list): dimensionality of the factors, list of integers with shape `(n,)` if the decomposition has `n`
			factors
			dmax (int): the maximal factor size of the decomposition

	Returns:
			bool: `True` if at least one couple of factors can be merged, `False` otherwise
	"""
	for i,ldi in enumerate(lens):
		if ldi < dmax:
			for ldj in lens[i+1:]:
				if ldj <= dmax - ldi:
					return True
	return False

def split_decomposition(decomposition, idx=None):
	"""Split a factor in the provided additive decomposition `decomposition`. If `idx` is provided, the split is performed
	on the factor of index `idx`, otherwise the factor is uniformly chosen among all the factors that can be split.

	Args:
			decomposition (list): a list of one numpy.array per factor of the decomposition, containing the dimension indices
			of the factor
			idx (int, optional): the index of the factor to split. Defaults to None.

	Returns:
			tuple: the decomposition with one factor split into two along with the probability of this outcome
	"""
	# List all the factors with their dimensionality
	idx_lens_no_filter = [(i,len(d)) for i,d in enumerate(decomposition)]
	# Filter the factors that can be split
	idx_lens = [(i,l) for i,l in idx_lens_no_filter if l > 1]

	# Sample a factor uniformingly if its index is not already provided
	prob_lvl1 = 1.0 / len(idx_lens)
	chosen_factor = idx_lens[np.random.randint(0, len(idx_lens))] if idx is None else idx_lens_no_filter[idx]
	prob_lvl2 = 2 ** (1 - chosen_factor[1])

	# Split the factor into two
	full_indices = np.arange(chosen_factor[1])
	chosen_seeds = np.array([0, 0])
	while chosen_seeds[0] == chosen_seeds[1]:
		chosen_seeds = np.random.choice(full_indices, size=(2,))

	set_flags = np.random.randint(0, 2, size=(chosen_factor[1],))
	set_flags[chosen_seeds[0]] = 0
	set_flags[chosen_seeds[1]] = 1

	# Rebuild the decomposition without the split
	decomposition_without_chosen = [d for i,d in enumerate(decomposition) if i != chosen_factor[0]]

	# Append the split, return the result
	chosen_splitted = [decomposition[chosen_factor[0]][set_flags == 0], decomposition[chosen_factor[0]][set_flags == 1]]
	return decomposition_without_chosen + chosen_splitted, prob_lvl1 * prob_lvl2

def merge_decomposition(decomposition, dmax):
	"""Merge a couple of factors in the provided additive decomposition `decomposition` while ensuring that the resulting
	decomposition contains only factors which are at most `dmax`-dimensional.

	Args:
			decomposition (list): a list of one numpy.array per factor of the decomposition, containing the dimension indices
			of the factor
			dmax (int): the maximal factor size of the decomposition

	Returns:
			tuple: the decomposition with a couple of factors merged along with the probability of this outcome
	"""
	# Choose a couple of factors uniformingly
	merge_couples = list_potential_merges([len(d) for d in decomposition], dmax)
	prob = 1.0 / len(merge_couples)
	idx_merge = np.random.randint(0, len(merge_couples))

	# Build the new decomposition
	decomposition_without_factors = [d for i,d in enumerate(decomposition) if i != merge_couples[idx_merge][0] and i != merge_couples[idx_merge][1]]
	merge = [np.concatenate((decomposition[merge_couples[idx_merge][0]], decomposition[merge_couples[idx_merge][1]]), axis=None)]
	return decomposition_without_factors + merge, prob

def sort_dimensions(dec):
	"""Sort the dimension indices in each factor of the provided additive decomposition `dec`.

	Args:
			dec (list): a list of one numpy.array per factor of the decomposition, containing the dimension indices of the
			factor

	Returns:
			list: the additive decomposition with each array of dimension indices sorted in ascending order
	"""
	return [np.sort(f) for f in dec]

def metropolis_hastings(decomposition, fitted_model, dmax, X, y, base_kernel_class, base_kernel_args):
	"""Sample a new additive decomposition based on the provided additive decomposition `decomposition` and the
	corresponding gaussian process `fitted_model`.

	Args:
			decomposition (list): a list of one numpy.array per factor of the decomposition, containing the dimension indices
			of the factor
			fitted_model (dumbo.optimize.MyOwnGP): the gaussian process built upon the decomposition `decomposition`
			dmax (int): the maximal factor size of the decomposition
			X (torch.Tensor): the training observations in the input space, shape `(n,d)` for `n` observations and
			`d`-dimensional input space
			y (torch.Tensor): the training observations in the output space, shape `(n,)` for `n` observations
			base_kernel_class (class from gpytorch.kernels.Kernel): the kernel used for each factor of the sampled additive
			decomposition.
			base_kernel_args (list): the arguments for the base kernel.

	Returns:
			tuple: the new sampled additive decomposition along with its corresponding gaussian process
	"""
	# Test for merging and splitting
	lens = np.array([len(d) for d in decomposition])
	split_authorized = np.any(lens > 1)
	merge_authorized = is_merge_allowed(lens, dmax)

	# If a new decomposition can be sampled
	if split_authorized or merge_authorized:
		choice = np.random.uniform()
		new_decomposition = None
		prob_new_given_actual = 0
		prob_actual_given_new = 0
		# Splitting
		if split_authorized and (choice < 0.5 or not merge_authorized):
			new_decomposition, prob = split_decomposition(decomposition)
			prob_new_given_actual = (0.5 if merge_authorized else 1.0) * prob
			_, prob_actual_given_new = merge_decomposition(new_decomposition, dmax)
		# Merging
		else:
			new_decomposition, prob = merge_decomposition(decomposition, dmax)
			prob_new_given_actual = (0.5 if split_authorized else 1.0) * prob
			_, prob_actual_given_new = split_decomposition(new_decomposition, idx=len(new_decomposition)-1)

		new_lens = np.array([len(d) for d in new_decomposition])
		new_split_authorized = np.any(new_lens > 1)
		new_merge_authorized = is_merge_allowed(new_lens, dmax)
		prob_actual_given_new *= 0.5 if new_split_authorized and new_merge_authorized else 1.0

		# Evaluate the likelihood of the old model
		likelihood_actual = fitted_model.get_mll()
		# Evaluate the likelihood of the new model
		new_model = dumbo.optimize.MyOwnGP.model_from_decomposition(new_decomposition, X, y, base_kernel_class=base_kernel_class, base_kernel_args=base_kernel_args)
		new_model.fit()
		likelihood_new = new_model.get_mll()

		# MH acceptance probability
		acceptance_prob = min(1.0, np.exp(likelihood_new + np.log(prob_actual_given_new) - likelihood_actual - np.log(prob_new_given_actual)))

		# If the new model is accepted, return it
		if np.random.uniform() < acceptance_prob:
			return sort_dimensions(new_decomposition), new_model

	# Otherwise, keep the old one
	return decomposition, fitted_model

def mcmc_sampling(decomposition, fitted_model, dmax, X, y, base_kernel_class, base_kernel_args, k=5):
	"""Sample `k` additive decompositions starting from the provided additive decomposition `decomposition`.

	Args:
			decomposition (list): a list of one numpy.array per factor of the decomposition, containing the dimension indices
			of the factor
			fitted_model (dumbo.optimize.MyOwnGP): the gaussian process built upon the decomposition `decomposition`
			dmax (int): the maximal factor size of the decomposition
			X (torch.Tensor): the training observations in the input space, shape `(n,d)` for `n` observations and
			`d`-dimensional input space
			y (torch.Tensor): the training observations in the output space, shape `(n,)` for `n` observations
			base_kernel_class (class from gpytorch.kernels.Kernel): the kernel used for each factor of the sampled additive
			decompositions.
			base_kernel_args (list): list of arguments for the base kernel.
			k (int, optional): the number of additive decompositions to sample. Defaults to 5.

	Returns:
			tuple: the list of sampled decompositions along with the gaussian processes built upon them
	"""
	models = [fitted_model]
	decompositions = [decomposition]

	for _ in range(k - 1):
		new_dec, new_model = metropolis_hastings(decompositions[-1], models[-1], dmax, X, y, base_kernel_class, base_kernel_args)
		models.append(new_model)
		decompositions.append(new_dec)

	return decompositions, models
