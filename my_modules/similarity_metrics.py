import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# === Data Analysis ===
def compute_histogram(image: np.ndarray) -> None:
    """Visualize Intensity Distribution"""
    sns.histplot(np.ravel(image), bins=256, color='skyblue')
    plt.title("Intensity Histogram")
    plt.xlabel("Pixel Intensity (0–255)")
    plt.ylabel("Frequency")
    plt.show()


# === Computation of Fitness Metrics ===
def compute_mi(fixed_image: np.ndarray, moving_image: np.ndarray, bins: int = 64) -> float:
    """Calculate Mutual Information (MI)"""
    fixed_image = fixed_image.ravel()
    moving_image = moving_image.ravel()

    hist_2d, _, _ = np.histogram2d(fixed_image, moving_image, bins=bins)
    p_xy = hist_2d / float(np.sum(hist_2d))  # joint probability
    p_x = np.sum(p_xy, axis=1)  # marginal probability  for x
    p_y = np.sum(p_xy, axis=0)  # marginal probability for y

    px_py = np.outer(p_x, p_y)  # joint distribution if x and y were statistically independent
    rows, cols = np.nonzero(p_xy)  # get non-zeros as log(0) is undefined
    mi = np.sum(p_xy[rows, cols] * np.log(p_xy[rows, cols] / px_py[rows, cols]))
    return mi


def compute_entropy(hist):
    """Calculate Entropy"""
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]  # Avoid log(0)
    return np.sum(prob * np.log(prob)) * (-1)


def compute_nmi(fixed_image: np.ndarray, moving_image: np.ndarray, bins: int = 64) -> float:
    """Calculate Normalized Mutual Information (based off of Studholme's definition)"""
    fixed_image = fixed_image.ravel()
    moving_image = moving_image.ravel()
    mask = np.isfinite(fixed_image) & np.isfinite(moving_image)
    moving_clean = moving_image[mask]  # removing NaN values from images
    fixed_clean = fixed_image[mask]

    hist_1, _ = np.histogram(fixed_clean, bins=bins, range=(0, 255))
    hist_2, _ = np.histogram(moving_clean, bins=bins, range=(0, 255))
    hist_2d, _, _ = np.histogram2d(fixed_clean, moving_clean, bins=bins, range=[[0, 255], [0, 255]])

    H1 = compute_entropy(hist_1)
    H2 = compute_entropy(hist_2)
    H12 = compute_entropy(hist_2d)

    nmi = (H1 + H2) / H12
    return nmi


# === Theoretical Fitness Metrics ====
def compute_sid(fixed_image: np.ndarray, moving_image: np.ndarray, bins: int = 256) -> float:
    """Compute Shared Information Distance (SID) between two images."""
    # Flatten images
    fixed_flat = fixed_image.ravel()
    moving_flat = moving_image.ravel()

    # Compute joint histogram
    joint_hist, _, _ = np.histogram2d(fixed_flat, moving_flat, bins=bins)
    joint_prob = joint_hist / np.sum(joint_hist)

    # Marginal probabilities
    p_x = np.sum(joint_prob, axis=1)
    p_y = np.sum(joint_prob, axis=0)

    # Entropies
    H_x = compute_entropy(p_x)
    H_y = compute_entropy(p_y)
    H_xy = compute_entropy(joint_prob)

    # Shared Information Distance
    sid = 2 * H_xy - H_x - H_y
    print("H_x:", H_x)
    print("H_y:", H_y)
    print("H_xy:", H_xy)
    print("SID:", sid)
    return sid


if __name__ == "__main__":
    ...