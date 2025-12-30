import numpy as np

# load result data
# order: [mi_before_registration, ground_truth_MI, score, mi_difference,translation_error_x, translation_error_y, rotation_error]
mri_1 = np.loadtxt("/Users/thien/Documents/Development/Image_Registration/results_1.csv", delimiter=",")
mri_2 = np.loadtxt("/Users/thien/Documents/Development/Image_Registration/results_2.csv", delimiter=",")
mri_3 = np.loadtxt("/Users/thien/Documents/Development/Image_Registration/results_3.csv", delimiter=",")
mri_4 = np.loadtxt("/Users/thien/Documents/Development/Image_Registration/results_4.csv", delimiter=",")
mri_5 = np.loadtxt("/Users/thien/Documents/Development/Image_Registration/results_5.csv", delimiter=",")

pet_1 = np.loadtxt("/Users/thien/Documents/Development/Image_Registration/results_pet_1.csv", delimiter=",")
pet_2 = np.loadtxt("/Users/thien/Documents/Development/Image_Registration/results_pet_2.csv", delimiter=",")
pet_3 = np.loadtxt("/Users/thien/Documents/Development/Image_Registration/results_pet_3.csv", delimiter=",")
pet_4 = np.loadtxt("/Users/thien/Documents/Development/Image_Registration/results_pet_4.csv", delimiter=",")
pet_5 = np.loadtxt("/Users/thien/Documents/Development/Image_Registration/results_pet_5.csv", delimiter=",")

# calculate averages
mri_averages = (mri_1 + mri_2 + mri_3 + mri_4 + mri_5) / 5.0
pet_averages = (pet_1 + pet_2 + pet_3 + pet_4 + pet_5) / 5.0

# calculate score standard deviation
all_scores = np.vstack((mri_1[:,2], mri_2[:,2], mri_3[:,2], mri_4[:,2], mri_5[:,2]))
std_scores = np.std(all_scores, axis=0)

print("finished")

