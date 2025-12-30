import time
import numpy as np
from my_modules import similarity_metrics, image_util
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


# === Metric Tracking ===
def compute_normalized_diversity(population: np.ndarray, min_bounds: np.ndarray, max_bounds:np.ndarray) -> float:
    """Average pairwise Euclidean distance in population normalized between 0 and 1."""
    # Compute average pairwise distance
    avg_distance = pdist(population).mean()

    # Maximum possible distance in this 3D space
    translation_range = max_bounds[0] - min_bounds[0]
    rotation_range = max_bounds[2] - min_bounds[2]
    max_distance = np.sqrt(translation_range ** 2 + translation_range ** 2 + rotation_range ** 2)

    # Normalize to [0,1]
    return avg_distance / max_distance


def convergence_metrics(population: np.ndarray, fitness: np.ndarray, mutation_strength: np.ndarray,
                        min_bounds: np.ndarray, max_bounds: np.ndarray) -> tuple:
    """Calculation of Convergence Metrics."""
    parameter_diversity = compute_normalized_diversity(population, min_bounds, max_bounds)
    fitness_std = np.std(fitness)
    fitness_mean = np.mean(fitness)
    fitness_best = np.max(fitness)
    translation_mutation_rate = mutation_strength[0]
    rotation_mutation_rate = mutation_strength[2]

    return parameter_diversity, fitness_std, fitness_mean, fitness_best, translation_mutation_rate, rotation_mutation_rate


# === Population Initialization ===
def initialize_population(size: int,
                          tx_bounds: tuple,
                          ty_bounds: tuple,
                          theta_bounds: tuple) -> np.ndarray:
    """Initialize population with individuals containing randomized transformation parameters."""
    translation_x = np.random.uniform(*tx_bounds, size=size)
    translation_y = np.random.uniform(*ty_bounds, size=size)
    rotation = np.random.uniform(*theta_bounds, size=size)  # rotation angle in degrees
    population = np.stack((translation_x, translation_y, rotation), axis=1)
    return population


def generate_individual(tx_bounds: tuple, ty_bounds: tuple, theta_bounds: tuple) -> np.ndarray:
    """Generate individual with randomized transformation parameters."""
    translation_x = np.random.uniform(*tx_bounds, size=1)
    translation_y = np.random.uniform(*ty_bounds, size=1)
    rotation = np.random.uniform(*theta_bounds, size=1)
    individual = np.stack((translation_x, translation_y, rotation), axis=1)
    return individual


# === Fitness Evaluation ===
# noinspection PyTypeChecker
def evaluate_fitness(fixed_image: np.ndarray,
                     moving_image: np.ndarray,
                     population: np.ndarray) -> np.ndarray:
    """Evaluate fitness of each individual."""
    fitness = np.empty(population.shape[0])
    for index, individual in enumerate(population):
        transformed_image = image_util.apply_affine_transform(image=moving_image,
                                                              tx=individual[0],
                                                              ty=individual[1],
                                                              theta=individual[2])
        fitness[index] = similarity_metrics.compute_mi(fixed_image, transformed_image)
    return fitness


def evaluate_fitness_p(fixed_image: np.ndarray,
                       moving_image: np.ndarray,
                       population: np.ndarray,
                       n_jobs: int = -1) -> np.ndarray:
    """Fitness evaluation where both image transformation and fitness computation are parallelized."""

    def evaluate_individual(individual):
        transformed_image = image_util.apply_affine_transform(moving_image,
                                                              tx=individual[0],
                                                              ty=individual[1],
                                                              theta=individual[2])
        return similarity_metrics.compute_mi(fixed_image, transformed_image)

    fitness = (Parallel(n_jobs=n_jobs)
               (delayed(evaluate_individual)(individual) for individual in population))

    return np.array(fitness)


def single_fitness(fixed_image: np.ndarray, moving_image: np.ndarray, individual: np.ndarray) -> np.ndarray:
    """Perform fitness evaluation on a single individual."""
    transformed_image = image_util.apply_affine_transform(moving_image,
                                                          tx=float(individual[0]),
                                                          ty=float(individual[1]),
                                                          theta=float(individual[2]))
    fitness = similarity_metrics.compute_mi(fixed_image, transformed_image)
    return np.array(fitness)


# === Survival Selection Methods ===
def roulette_selection(population: np.ndarray,
                       fitness: np.ndarray,
                       num_selected: int) -> np.ndarray:
    """Roulette Wheel Selection, where chance of survival is entirely dependent on relative fitness scores."""
    probabilities = fitness / fitness.sum()
    indices = np.random.choice(a=len(population), size=num_selected, p=probabilities)
    return population[indices]


def tournament_selection(population: np.ndarray,
                         fitness: np.ndarray,
                         num_selected: int,
                         tournament_size: int = 3) -> tuple:
    """Tournament Selection, where random individuals enter tournament where the fittest survives."""
    n_individuals = population.shape[0]
    selected_indices = np.zeros(num_selected, dtype=int)

    for i in range(num_selected):
        # Select random individuals and get their similarity scores
        tournament_indices = np.random.choice(a=n_individuals, size=tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]

        # Select individual with the highest fitness scores
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_indices[i] = winner_index

    selected_population = population[selected_indices]
    return selected_population, selected_indices


def elitism_selection(population: np.ndarray, fitness: np.ndarray, n: int = 2) -> tuple:
    """Select the n best amount of individuals according to fitness."""
    best_indices = np.argsort(fitness)[-n:]
    best_individuals = population[best_indices]
    return best_individuals, best_indices


def single_elite(population: np.ndarray, fitness: np.ndarray) -> tuple:
    """Select one elite to carry over to the next generation."""
    best_index = np.argmax(fitness)
    best_individual = population[best_index]
    return best_individual, best_index


# === Crossover Methods ===
def select_parent_indices(selected_pop: np.ndarray, target_size: int) -> tuple:
    """Randomly select parent indices indicating which parents will produce offspring"""
    if target_size % 2 != 0:
        target_size += 1

    # randomly select indices of individuals who will mate
    n_selected = selected_pop.shape[0]
    parent_pairs = np.random.choice(n_selected,
                                    size=(target_size // 2, 2),
                                    replace=True)
    parent_indices_1 = parent_pairs[:, 0]
    parent_indices_2 = parent_pairs[:, 1]

    # Retrieve parent pairs
    parents1 = selected_pop[parent_indices_1]  # shape: (n_pairs, n_params)
    parents2 = selected_pop[parent_indices_2]  # shape: (n_pairs, n_params)
    return parents1, parents2

def arithmetic_crossover(selected_pop: np.ndarray,
                         target_size: int) -> np.ndarray:
    """Arithmetic Crossover, where the child is the weighted average of the parents."""
    # Get parent indices
    parents1, parents2 = select_parent_indices(selected_pop, target_size)

    # Randomize alphas and creating children
    alphas1 = np.random.uniform(0.1, 0.9, size=(target_size // 2, 1))
    alphas2 = np.random.uniform(0.1, 0.9, size=(target_size // 2, 1))

    child1 = alphas1 * parents1 + (1 - alphas1) * parents2
    child2 = alphas2 * parents2 + (1 - alphas2) * parents1

    # Stack all offspring and trim to exact target size
    offspring = np.vstack((child1, child2))[:target_size]
    return offspring


def blend_crossover(selected_pop: np.ndarray,
                    target_size: int,
                    alpha: float = 0.5) -> np.ndarray:
    """Blend Crossover, where children are sampled from an extended range around the parents' values."""
    # Get parent indices
    parents1, parents2 = select_parent_indices(selected_pop=selected_pop, target_size=target_size)

    # Calculate the range between parents for each parameter
    d = np.abs(parents1 - parents2)  # shape: (n_pairs, n_params)

    # Define the extended interval [interval_min, interval_max] for each parameter
    interval_min = np.minimum(parents1, parents2) - alpha * d
    interval_max = np.maximum(parents1, parents2) + alpha * d

    # Sample children uniformly from the extended intervals
    child1 = np.random.uniform(interval_min, interval_max)
    child2 = np.random.uniform(interval_min, interval_max)

    # Stack all offspring and trim to exact target size
    offspring = np.vstack((child1, child2))[:target_size]
    return offspring


# === Mutation Methods ===
def gaussian_mutation(population: np.ndarray,
                      mutation_rate: np.ndarray,
                      mins: np.ndarray,
                      maxs: np.ndarray) -> np.ndarray:
    """Gaussian Mutation where gaussian noise is added onto existing individuals."""
    n_individuals, n_params = population.shape
    mutated_pop = population.copy()

    # Generate Gaussian noise for all parameters (vectorized)
    noise = np.random.normal(0, mutation_rate, (n_individuals, n_params))
    # Add gaussian noise to population
    mutated_pop = mutated_pop + noise

    # Clip to bounds
    mutated_pop = np.clip(mutated_pop, mins, maxs)

    return mutated_pop


def linear_decrease_mutation(initial_rate: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
    """Linearly decrease mutation rate over generations."""
    return initial_rate * (1 - generation/ max_generations)


def exponential_decrease_mutation(initial_rate: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
    """Exponentially decrease mutation rate over generations."""
    k = 1.0 / max_generations
    return initial_rate * np.exp(-k * generation)


# == Genetic Algorithm Designs
def mu_plus_lambda(fixed: np.ndarray,
                   moving: np.ndarray,
                   population_size: int,
                   max_generations: int,
                   minimums: np.ndarray,
                   maximums: np.ndarray,
                   initial_mutation_rates: np.ndarray) -> tuple[np.ndarray, ...]:
    """Genetic Algorithm with mu+lambda strategy."""
    # First generation
    population = initialize_population(size=population_size,
                                       tx_bounds=(minimums[0], maximums[0]),
                                       ty_bounds=(minimums[1], maximums[1]),
                                       theta_bounds=(minimums[2], maximums[2]))
    fitness = evaluate_fitness_p(fixed_image=fixed, moving_image=moving, population=population)
    new_mutation_rates = initial_mutation_rates.copy()

    # Storing convergence metrics
    parameter_diversity = np.empty(shape=(max_generations+1,))
    fitness_std = np.empty(shape=(max_generations+1,))
    fitness_mean = np.empty(shape=(max_generations+1,))
    fitness_best = np.empty(shape=(max_generations+1,))
    translation_mutation_rate = np.empty(shape=(max_generations+1,))
    rotation_mutation_rate = np.empty(shape=(max_generations+1,))

    # Main GA loop
    for generation in range(max_generations):
        # Recording Metrics
        (p_diversity, fit_std, fit_mean, fit_best,
         t_mut_rate, rot_mut_rate) = convergence_metrics(population=population,
                                                         fitness=fitness,
                                                         mutation_strength=new_mutation_rates,
                                                         min_bounds=minimums,
                                                         max_bounds=maximums)
        parameter_diversity[generation] = p_diversity
        fitness_std[generation] = fit_std
        fitness_mean[generation] = fit_mean
        fitness_best[generation] = fit_best
        translation_mutation_rate[generation] = t_mut_rate
        rotation_mutation_rate[generation] = rot_mut_rate

        # Parent selection
        parents = tournament_selection(population=population,
                                       fitness=fitness,
                                       num_selected=population_size,
                                       tournament_size=2)[0]

        # Offspring creation
        offspring = blend_crossover(selected_pop=parents, target_size=2*population_size, alpha=0.5)

        # Offspring mutation
        new_mutation_rates = linear_decrease_mutation(initial_rate=initial_mutation_rates,
                                                      generation=generation+1,
                                                      max_generations=max_generations+1)

        offspring = gaussian_mutation(population = offspring,
                                      mutation_rate=new_mutation_rates,
                                      mins=minimums,
                                      maxs=maximums)

        # Offspring fitness calculation
        offspring_fitness = evaluate_fitness_p(fixed_image=fixed,
                                               moving_image=moving,
                                               population=offspring)

        # Combine populations
        combined_pop = np.vstack([population, offspring])
        combined_fitness = np.concatenate([fitness, offspring_fitness])

        # Select best mu individuals
        best_indices = np.argsort(combined_fitness)[-population_size:]
        population = combined_pop[best_indices]
        fitness = combined_fitness[best_indices]

    # Get optimal transformation parameters for transformation of moving image
    best_index = np.argmax(fitness)
    best_individual = population[best_index]
    best_score = fitness[best_index]

    # Recording Metrics for the last generation
    (p_diversity, fit_std, fit_mean, fit_best,
     t_mut_rate, rot_mut_rate) = convergence_metrics(population=population,
                                                     fitness=fitness,
                                                     mutation_strength=new_mutation_rates,
                                                     min_bounds=minimums,
                                                     max_bounds=maximums)
    parameter_diversity[max_generations] = p_diversity
    fitness_std[max_generations] = fit_std
    fitness_mean[max_generations] = fit_mean
    fitness_best[max_generations] = fit_best
    translation_mutation_rate[max_generations] = t_mut_rate
    rotation_mutation_rate[max_generations] = rot_mut_rate

    # Plotting convergence metrics
    fig, axs = plt.subplots(3, 2, figsize=(12, 14))
    axs = axs.ravel()

    axs[0].plot(parameter_diversity, label="Parameter Diversity", color="purple")
    axs[0].set_title("Parameter Diversity")
    axs[0].set_ylabel("Diversity")
    axs[0].set_xlabel("Generation")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(fitness_std, label="Fitness Std", color="orange")
    axs[1].set_title("Fitness Standard Deviation")
    axs[1].set_ylabel("Std")
    axs[1].set_xlabel("Generation")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(fitness_mean, label="Fitness Mean", color="blue")
    axs[2].set_title("Fitness Mean")
    axs[2].set_ylabel("Mean Fitness")
    axs[2].set_xlabel("Generation")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(fitness_best, label="Fitness Best", color="green")
    axs[3].set_title("Best Fitness")
    axs[3].set_ylabel("Best Fitness")
    axs[3].set_xlabel("Generation")
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)

    axs[4].plot(translation_mutation_rate, label="Translation Mutation Rate", color="red")
    axs[4].set_title("Translation Mutation Rate (Adaptive)")
    axs[4].set_ylabel("Rate")
    axs[4].set_xlabel("Generation")
    axs[4].legend()
    axs[4].grid(True, alpha=0.3)

    axs[5].plot(rotation_mutation_rate, label="Rotation Mutation Rate", color="brown")
    axs[5].set_title("Rotation Mutation Rate (Adaptive)")
    axs[5].set_ylabel("Rate")
    axs[5].set_xlabel("Generation")
    axs[5].legend()
    axs[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()

    return best_individual, best_score


# === Main Function ===
def main():
    # Configuration
    pop_size = 100
    max_gens = 30
    mins = np.array([-75.0, -75.0, -25.0])
    maxs = np.array([75.0, 75.0, 25.0])
    initial_mutation = np.array([30, 30, 10])  # equal to 20% of parameter range

    # for saving experimental results
    results = np.zeros(shape=(18,7))

    # load ground_truth transformation parameters
    ground_truth = np.loadtxt(
        "/Users/thien/Documents/Development/Image_Registration/dataset/ct_t1_ground_truth/transformations.csv",
        delimiter=",")
    ground_truth = ground_truth * (-1)

    for patient_number in range(18, 19):
        ground_truth_index = patient_number - 1

        # Running the algorithm
        ct = image_util.read_img(f"./dataset/ct_t1_test/CT_Patient({patient_number}).png")
        t1 = image_util.read_img(f"./dataset/ct_t1_test/T1_Patient({patient_number}).png")
        mi_before_registration = similarity_metrics.compute_mi(ct,t1)

        start = time.perf_counter()
        params, score = mu_plus_lambda(fixed=ct,
                                       moving=t1,
                                       population_size=pop_size,
                                       max_generations=max_gens,
                                       minimums=mins,
                                       maximums=maxs,
                                       initial_mutation_rates=initial_mutation)
        end = time.perf_counter()
        print(f"Total time: {end - start} \n")

        # show registration results
        image_util.visualize_registration_comparison(ct, t1)
        moving_transformed = image_util.apply_affine_transform(image=t1,
                                                               tx=float(params[0]),
                                                               ty=float(params[1]),
                                                               theta=float(params[2]))
        image_util.visualize_registration_comparison(ct, moving_transformed)
        # image_util.fuse_images_gray(ct, moving_transformed)

        # load ground_truth images
        ct_truth = image_util.read_img(f"./dataset/ct_t1_ground_truth/CT_Patient({patient_number}).png")
        mri_truth = image_util.read_img(f"./dataset/ct_t1_ground_truth/T1_Patient({patient_number}).png")

        # calculate registration error metrics
        translation_error_x = np.abs(ground_truth[ground_truth_index, 0] - params[0])
        translation_error_y = np.abs(ground_truth[ground_truth_index, 1] - params[1])

        theta_ground_truth = np.deg2rad(ground_truth[ground_truth_index, 2])
        theta_estimated = np.deg2rad(params[2])
        rotation_error = np.rad2deg(np.arctan2(np.sin(theta_ground_truth - theta_estimated),
                                    np.cos(theta_ground_truth - theta_estimated)))

        # calculate differences in MI scores
        ground_truth_MI = similarity_metrics.compute_mi(ct_truth, mri_truth)
        mi_difference = ground_truth_MI - score

        print(f"mi_before_registration: {mi_before_registration} \n"
              f"ground_truth_MI: {ground_truth_MI} \n"
              f"mi_after_registration: {score} \n"
              f"mi_difference: {mi_difference} \n"
              f"translation_error_x: {translation_error_x} \n"
              f"translation_error_y: {translation_error_y} \n"
              f"rotation_error: {rotation_error} \n")

        results[patient_number-1] = [mi_before_registration, ground_truth_MI, score, mi_difference,translation_error_x, translation_error_y, rotation_error]

    np.savetxt(fname="results.csv", X=results, delimiter=",")

def ct_pet_registration():
    # Configuration
    pop_size = 100
    max_gens = 30
    mins = np.array([-75.0, -75.0, -25.0])
    maxs = np.array([75.0, 75.0, 25.0])
    initial_mutation = np.array([30, 30, 10])  # equal to 20% of parameter range

    results = np.zeros(shape=(3,2))

    for patient_number in range(1, 4):
        # Running the algorithm
        ct = image_util.read_img(f"./dataset/ct_pet/ct_{patient_number}.png")
        pet = image_util.read_img(f"./dataset/ct_pet/pet_{patient_number}.png")

        mi_before_registration = similarity_metrics.compute_mi(ct,pet)

        start = time.perf_counter()
        params, score = mu_plus_lambda(fixed=ct,
                                       moving=pet,
                                       population_size=pop_size,
                                       max_generations=max_gens,
                                       minimums=mins,
                                       maximums=maxs,
                                       initial_mutation_rates=initial_mutation)
        end = time.perf_counter()
        print(f"Total time: {end - start} \n")

        # show registration results
        ct_bright = image_util.equalize_hist(ct)
        pet_bright = image_util.equalize_hist(pet,4)
        # image_util.visualize_registration_comparison(ct_bright, pet_bright)
        image_util.fuse_ct_pet(ct_bright, pet_bright)
        moving_transformed = image_util.apply_affine_transform(image=pet_bright,
                                                               tx=float(params[0]),
                                                               ty=float(params[1]),
                                                               theta=float(params[2]))

        mi_after_registration = similarity_metrics.compute_mi(ct,moving_transformed)
        results[patient_number-1] = [mi_before_registration, mi_after_registration]

        # image_util.visualize_registration_comparison(ct_bright, moving_transformed)
        moving_hot = image_util.equalize_hist(moving_transformed,1)
        image_util.fuse_ct_pet(ct, moving_hot)
    np.savetxt("results_pet_3.csv", results, delimiter=",")


# === Genetic Algorithms ===
if __name__ == "__main__":
    ct_pet_registration()
