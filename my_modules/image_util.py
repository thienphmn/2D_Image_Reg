import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import skimage as sk
from matplotlib.colors import ListedColormap
import matplotlib

# === Reading Images ===
def read_img(full_path: str, downsample_size:int=1) -> np.ndarray:
    """Read common image file types and convert to grayscale."""
    # read image with according downsampling method
    full_path = full_path
    if downsample_size == 1:
        array = cv2.imread(filename=full_path, flags=cv2.IMREAD_GRAYSCALE)
    elif downsample_size == 2:
        array = cv2.imread(filename=full_path, flags=cv2.IMREAD_REDUCED_GRAYSCALE_2)
    elif downsample_size == 4:
        array = cv2.imread(filename=full_path, flags=cv2.IMREAD_REDUCED_COLOR_4)
    elif downsample_size == 8:
        array = cv2.imread(filename=full_path, flags=cv2.IMREAD_REDUCED_GRAYSCALE_8)
    else:
        raise ValueError('scale must be 1, 2, 4 or 8')

    # apply preprocessing
    array = gaussian_blurr(array)
    array = normalize_img(array)
    return array


def read_dcm(full_path: str, window: bool = False, upsample:bool=False) -> np.ndarray:
    """Read DICOM file and convert pixel intensities to uint8 grayscale."""
    # Load image using SimpleITK
    img = sitk.ReadImage(full_path)
    array = sitk.GetArrayFromImage(img)[0]  # Extract 2D slice from volume

    # upsample PET to match CT dimensions
    if upsample:
        array = cv2.resize(array, (512, 512), interpolation=cv2.INTER_CUBIC)

    # apply soft tissue window to CT scan
    if window:
        array = apply_window(array, level=40, width=400)

    array = array.astype(np.float32)
    array -= np.min(array)
    array /= np.max(array)
    array *= 255.0
    array = array.astype(np.uint8)

    # Optional preprocessing
    array_blurred = gaussian_blurr(array)
    array_processed = normalize_img(array_blurred)

    return array_processed

def apply_window(image: np.ndarray, level: int, width: int) -> np.ndarray:
    """Apply windowing to CT image and convert to uint8."""
    lower = level - (width // 2)
    upper = level + (width // 2)
    windowed = np.clip(image, lower, upper)
    windowed = ((windowed - lower) / (upper - lower)) * 255.0
    return windowed.astype(np.uint8)


# === Image Preprocessing ===
def normalize_img(image:np.ndarray) -> np.ndarray:
    """Normalize the image, so intensities take up complete intensity spectrum."""
    normalized_image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized_image

def equalize_hist(image: np.ndarray, clip_limit: float = 4.0) -> np.ndarray:
    """Equalize the histogram."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced

def gaussian_blurr(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian Blur to reduce noise."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred


def pad_image(image: np.ndarray, pad_size: int, value=0) -> np.ndarray:
    """Add equal padding on all sides of an image."""
    return np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)),
                  mode='constant', constant_values=(value, value))


# === Show Images ===
def show_image(image: np.ndarray) -> None:
    """Show single image as grayscale."""
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.close()


def compare_images(image_1: np.ndarray, image_2: np.ndarray, method: str = "checkerboard",
                   tiles: tuple = (2, 2)) -> None:
    """Checkerboard image comparison."""
    comparison = sk.util.compare_images(image_1, image_2, method=method, n_tiles=tiles)
    plt.imshow(comparison, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.close()


def fuse_images_rgb(image_1: np.ndarray, image_2: np.ndarray) -> None:
    """Image overlay where one image is green while the other one is magenta."""
    fused = np.stack([image_2, image_1, image_2], axis=2)
    plt.imshow(fused)
    plt.axis('off')
    plt.show()
    plt.close()


def fuse_images_gray(image_1: np.ndarray, image_2: np.ndarray) -> None:
    """Simple image fusion with control over the alpha of the image on top."""
    for i in range(11):
        alpha = 0.1 * i
        plt.imshow(image_1, cmap='gray', alpha=1)
        plt.imshow(image_2, cmap='gray', alpha=alpha)
        plt.axis('off')
        plt.show()
        plt.close()


def visualize_registration_comparison(image_1: np.ndarray, image_2: np.ndarray,
                                      method: str = "checkerboard", tiles: tuple = (2, 2)) -> None:
    """Displays checkerboard and green-magenta fusion side by side."""

    # Generate checkerboard comparison
    checkerboard = sk.util.compare_images(image_1, image_2, method=method, n_tiles=tiles)

    # Generate green-magenta fusion
    fused = np.stack([image_2, image_1, image_2], axis=2)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(checkerboard, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(fused)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


def fuse_ct_pet(ct: np.ndarray, pet: np.ndarray) -> None:
    """Fuse ct and pet, where ct is grayscale and pet is a hot colormap."""
    # Overlay CT and PET image with sigmoid transparency applied to PET
    plt.imshow(ct, cmap="gray", vmin=0, vmax=255)
    plt.imshow(pet, cmap="turbo", vmin=0, vmax=255, alpha=0.5)
    plt.axis('off')
    plt.show()


# === Image Transformation and Masking ===
def apply_affine_transform(image: np.ndarray, tx: float = 0, ty: float = 0,
                           theta: float = 0, scale: float = 1.0) -> np.ndarray:
    """Apply affine transformation to an image using OpenCV, rotating and scaling around the center."""
    cv2.setNumThreads(1)
    H, W = image.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # Convert to radians and compute rotation terms
    a = np.deg2rad(theta)
    cos_a, sin_a = np.cos(a), np.sin(a)

    # Compute the affine matrix directly -> Translate to center; rotate and scale; translate back; user translation
    M = np.array([
        [scale * cos_a, -scale * sin_a, tx + cx * (1 - scale * cos_a) + cy * (scale * sin_a)],
        [scale * sin_a, scale * cos_a, ty + cy * (1 - scale * cos_a) - cx * (scale * sin_a)]
    ], dtype=np.float32)

    # Apply the transformation
    return cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR)


def combined_foreground(image_1: np.ndarray, image_2: np.ndarray) -> np.ndarray:
    """Create a binary mask for removing background in both images"""
    _, mask_fixed = cv2.threshold(image_1, 10, 255, cv2.THRESH_BINARY)
    _, mask_moving = cv2.threshold(image_2, 10, 255, cv2.THRESH_BINARY)

    # Convert to binary masks (0 or 1)
    mask_fixed = (mask_fixed > 0).astype(np.uint8)
    mask_moving = (mask_moving > 0).astype(np.uint8)

    # Combine masks: keep only pixels that are foreground in both
    combined_mask = cv2.bitwise_and(mask_fixed, mask_moving)

    return combined_mask


def binary_filter(image: np.ndarray) -> np.ndarray:
    """Create binary image, where foreground is white and background is black."""
    ret, binary_img = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    return binary_img


# == Create synthetic perturbations for benchmarking ===
def synthetic_perturbations() -> None:
    # set boundaries
    min_bounds = np.array([-50.0, -15.0])
    max_bounds = np.array([50.0, 15.0])

    # initialize empty array in which the random transformations are saved
    transformations = np.empty(shape=(18, 3))

    for patient_number in range(1, 19):
        # load images
        fixed = read_img(
            f"/Users/thien/Documents/Development/Image_Registration/dataset/ground_truth/CT_Patient({patient_number}).png")
        moving = read_img(
            f"/Users/thien/Documents/Development/Image_Registration/dataset/ground_truth/T1_Patient({patient_number}).png")

        # pad images, to avoid clipping after transformations
        fixed = pad_image(fixed, 100, 0)
        moving = pad_image(moving, 100, 0)

        # random perturbations
        tx = random.uniform(float(min_bounds[0]), float(max_bounds[0]))
        ty = random.uniform(float(min_bounds[0]), float(max_bounds[0]))
        theta = random.uniform(float(min_bounds[1]), float(max_bounds[1]))

        # save perturbations to array
        transformations[patient_number - 1] = tx, ty, theta
        print(tx, ty, theta)

        # apply transformations
        moving_t = apply_affine_transform(image=moving, tx=tx, ty=ty, theta=theta, scale=1)
        show_image(moving_t)

        # save fixed and moving images
        cv2.imwrite(f"CT_Patient({patient_number}).png", fixed)
        cv2.imwrite(f"T1_Patient({patient_number}).png", moving_t)

    # save array with perturbations to .csv file
    np.savetxt("transformations.csv", transformations, delimiter=",")


if __name__ == '__main__':
    pet = read_img("/Users/thien/Documents/Development/Image_Registration/dataset/ct_pet/pet_1.png")
    pet = equalize_hist(pet, 1)
    show_image(pet)




