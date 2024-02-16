# Prevent NumPy from going crazy on multi-threading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from dumbo.optimize import DuMBOOptimizer
import numpy as np

# Objective function intervals
def hartmann6d_intervals():
	return np.concatenate((np.zeros((6,1)), np.ones((6,1))), axis=1)

# Objective function
def hartmann6d(x):
	alpha = np.array([1.0, 1.2, 3.0, 3.2])
	A = np.array([
		[10, 3, 17, 3.5, 1.7, 8],
		[0.05, 10, 17, 0.1, 8, 14],
		[3, 3.5, 1.7, 10, 17, 8],
		[17, 8, 0.05, 10, 0.1, 14]
	])
	P = np.array([
		[1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
		[2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
		[2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
		[4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]
	]) / 1e4

	f = 0
	for i in range(4):
		f += alpha[i] * np.exp(-np.dot(A[i], np.power(x - P[i], 2)))

	return f

if __name__ == "__main__":
	max_objective_function_value = 3.32237
	intervals = hartmann6d_intervals()
	# Create the optimizer, set the hyperparameters
	optimizer = DuMBOOptimizer(intervals, n_init_points=2, n_samples_per_iteration=5, uniform_decomposition_sampling=False, max_it=8)

	# Simple optimization loop
	n_iterations = 150
	regret = np.array([])
	for i in range(n_iterations):
		# Query the next point
		next_query = optimizer.next_query()
		# Collect the objective function value at this point
		objective_function_value = hartmann6d(next_query)
		# Add an observation to the optimizer
		optimizer.tell(next_query, objective_function_value)

		# Save the result and print simple information
		regret = np.concatenate((regret, np.array([max_objective_function_value - objective_function_value])))
		print(f"== Step {i+1} ==")
		print(f"* Regret: {np.round(regret[-1], 5)} \t* AvgRegret: {np.round(np.average(regret), 5)} \t* MinRegret: {np.round(np.min(regret), 5)} \t* x: {np.round(next_query, 5)}")