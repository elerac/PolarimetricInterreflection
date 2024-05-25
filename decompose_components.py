import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import cv2

import polanalyser as pa


def decompose_by_polarization_rotation(image_list: npt.ArrayLike, polarizer_angles_psg: List[float], polarizer_angles_psa: List[float]) -> Tuple[np.ndarray]:
    """Decompose the specular inter-reflection components by analyzing the rotation direction of linear polarization.

    Parameters
    ----------
    imlist : List[np.ndarray]
        Captured images
    polarizer_angles_psg : List[float]
        Polarizer angles on the light side, in radian
    polarizer_angles_psa : List[float]
        Polarizer angles on the detector (camera) side, in radian

    Returns
    -------
    intensity_forward, phase_forward : np.ndarray
        Forward rotation component
    intensity_reverse, phase_reverse : np.ndarray
        Reverse rotation component
    intensity_unpolarized : np.ndarray
        Unpolarized component
    """
    # From intensity images (identical to our paper)
    intensities = np.stack(image_list, axis=-1).astype(np.float64)  # (height, width, N)
    polarizer_angles_psg = np.stack(polarizer_angles_psg, axis=-1).astype(np.float64)  # (N,)
    polarizer_angles_psa = np.stack(polarizer_angles_psa, axis=-1).astype(np.float64)  # (N,)
    ones = np.ones_like(polarizer_angles_psg)
    cos_psg = np.cos(2 * polarizer_angles_psg)
    sin_psg = np.sin(2 * polarizer_angles_psg)
    cos_psa = np.cos(2 * polarizer_angles_psa)
    sin_psa = np.sin(2 * polarizer_angles_psa)
    W = np.array([ones, cos_psa * cos_psg, sin_psa * sin_psg, cos_psa * sin_psg, sin_psa * cos_psg]).T  # (N, 5)

    # From Mueller matrix
    # mueller_psa_list = [2 * pa.polarizer(rad)[:3, :3] for rad in polarizer_angles_psa]
    # mueller_psg_list = [2 * pa.polarizer(rad)[:3, :3] for rad in polarizer_angles_psg]
    # img_mueller = pa.calcMueller(image_list, mueller_psg_list, mueller_psa_list)
    # x1 = img_mueller[..., 0, 0]  # m00
    # x2 = img_mueller[..., 1, 1]  # m11
    # x5 = img_mueller[..., 1, 2]  # m12
    # x4 = img_mueller[..., 2, 1]  # m21
    # x3 = img_mueller[..., 2, 2]  # m22

    rank = np.linalg.matrix_rank(W)
    if rank < 5:
        raise ValueError(f"Rank of W is {rank} < 5.")

    # cond = np.linalg.cond(W)

    W_pinv = np.linalg.pinv(W)  # (5, N)
    X = np.tensordot(W_pinv, intensities, axes=(1, -1))  # (5, height, width)
    x1, x2, x3, x4, x5 = X

    # Forward rotation component
    intensity_forward = np.sqrt((x2 - x3) ** 2 + (x4 + x5) ** 2)  # (height, width)
    phase_forward = 0.5 * np.arctan2(x4 + x5, x2 - x3)  # (height, width)

    # Reverse rotation component
    intensity_reverse = np.sqrt((x2 + x3) ** 2 + (-x4 + x5) ** 2)  # (height, width)
    phase_reverse = 0.5 * np.arctan2(-x4 + x5, x2 + x3)  # (height, width)

    # Unpolarized component
    intensity_unpolarized = 2.0 * x1 - (intensity_forward + intensity_reverse)  # (height, width)

    return intensity_forward, phase_forward, intensity_reverse, phase_reverse, intensity_unpolarized


def adjust_image(image: np.ndarray, gamma_enable: bool = True) -> np.ndarray:
    """Adjust the image for visualization."""
    image = np.clip(image, 0, 1)
    if gamma_enable:
        image = np.where(image <= 0.0031308, 12.92 * image, 1.055 * image ** (1 / 2.4) - 0.055)
    return np.clip(255.0 * image, 0, 255).astype(np.uint8)


def main():
    warnings.simplefilter("ignore", RuntimeWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input data path")
    parser.add_argument("--intensity_scale", type=float, default=1.0, help="Intensity scale for visualization")
    parser.add_argument("-o", "--output", type=Path, help="Output path")
    args = parser.parse_args()
    path_src = args.input
    path_dst = args.output

    # Load images
    print(f"Load polarization images from '{path_src}'.")
    pcontainer = pa.PolarizationContainer(path_src)
    images = pcontainer.get_list("image")
    images = [np.clip(img, 0, None) for img in images]
    polarizer_angles_psg = np.deg2rad(pcontainer.get_list("polarizer_angle_psg"))
    polarizer_angles_psa = np.deg2rad(pcontainer.get_list("polarizer_angle_psa"))

    is_color = images[0].ndim == 3

    # Decompose
    (
        intensity_forward,
        phase_forward,
        intensity_reverse,
        phase_reverse,
        intensity_unpolarized,
    ) = decompose_by_polarization_rotation(images, polarizer_angles_psg, polarizer_angles_psa)

    intensity_forward *= args.intensity_scale
    intensity_reverse *= args.intensity_scale
    intensity_unpolarized *= args.intensity_scale

    img_all = intensity_forward + intensity_reverse + intensity_unpolarized

    # Distribution of three components
    img_distribution = np.stack([intensity_reverse, intensity_forward, intensity_unpolarized], axis=-1)
    if is_color:
        img_distribution = np.average(img_distribution, axis=-2)

    img_distribution_u8 = adjust_image(img_distribution / 2)
    img_distribution_normalized = img_distribution / np.sum(img_distribution, axis=-1)[..., None]
    img_distribution_normalized_u8 = adjust_image(img_distribution_normalized, gamma_enable=False)

    # Forward
    intensity_forward_u8 = adjust_image(intensity_forward)
    if is_color:
        phase_forward = np.average(phase_forward, axis=-1)
    phase_forward_u8 = pa.applyColorToAoLP(phase_forward, value=img_distribution_normalized[..., 1])

    # Reverse
    intensity_reverse_u8 = adjust_image(intensity_reverse)
    if is_color:
        phase_reverse = np.average(phase_reverse, axis=-1)
    phase_reverse_u8 = pa.applyColorToAoLP(phase_reverse, value=img_distribution_normalized[..., 0])

    # Unpolarized
    intensity_unpolarized_u8 = adjust_image(intensity_unpolarized)

    # All
    img_all_u8 = adjust_image(img_all)

    # Export images
    if path_dst is None:
        path_dst = Path(path_src) / Path("decomposition_results")
    print(f"Export decomposed results to '{path_dst}'.")
    path_dst.mkdir(parents=True, exist_ok=True)

    # PNG
    cv2.imwrite(f"{path_dst}/intensity_forward.png", intensity_forward_u8)
    cv2.imwrite(f"{path_dst}/intensity_reverse.png", intensity_reverse_u8)
    cv2.imwrite(f"{path_dst}/phase_forward.png", phase_forward_u8)
    cv2.imwrite(f"{path_dst}/phase_reverse.png", phase_reverse_u8)
    cv2.imwrite(f"{path_dst}/intensity_unpolarized.png", intensity_unpolarized_u8)
    cv2.imwrite(f"{path_dst}/intensity_all.png", img_all_u8)
    cv2.imwrite(f"{path_dst}/distribution.png", img_distribution_u8)
    cv2.imwrite(f"{path_dst}/distribution_norm.png", img_distribution_normalized_u8)

    # EXR
    cv2.imwrite(f"{path_dst}/intensity_forward.exr", intensity_forward.astype(np.float32))
    cv2.imwrite(f"{path_dst}/intensity_reverse.exr", intensity_reverse.astype(np.float32))
    cv2.imwrite(f"{path_dst}/phase_forward.exr", phase_forward.astype(np.float32))
    cv2.imwrite(f"{path_dst}/phase_reverse.exr", phase_reverse.astype(np.float32))
    cv2.imwrite(f"{path_dst}/intensity_unpolarized.exr", intensity_unpolarized.astype(np.float32))
    cv2.imwrite(f"{path_dst}/intensity_all.exr", img_all.astype(np.float32))


if __name__ == "__main__":
    main()
