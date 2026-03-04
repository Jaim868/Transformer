from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticQSMConfig:
    image_size: int = 128
    lesions_per_image: tuple[int, int] = (1, 4)
    susceptibility_range: tuple[float, float] = (0.02, 0.25)
    noise_std: float = 0.03
    seed: int = 42


class SyntheticQSMDataset(Dataset):
    """Synthetic QSM dataset for algorithm prototyping and model comparison.

    It emulates an ill-posed inverse mapping from local field perturbation to
    susceptibility map using a simplified dipole-like blur in k-space.
    """

    def __init__(self, length: int, config: SyntheticQSMConfig):
        self.length = length
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        grid = np.linspace(-1.0, 1.0, config.image_size, dtype=np.float32)
        xx, yy = np.meshgrid(grid, grid, indexing="ij")
        self.radius_template = np.sqrt(xx**2 + yy**2)
        kx, ky = np.meshgrid(
            np.fft.fftfreq(config.image_size),
            np.fft.fftfreq(config.image_size),
            indexing="ij",
        )
        dipole_kernel = 1.0 / (1.0 + 20.0 * (kx**2 + ky**2))
        dipole_kernel[0, 0] = 0.0
        self.dipole_kernel = dipole_kernel.astype(np.float32)

    def __len__(self) -> int:
        return self.length

    def _generate_susceptibility(self) -> np.ndarray:
        susceptibility = np.zeros((self.config.image_size, self.config.image_size), dtype=np.float32)
        n_lesions = self.rng.integers(self.config.lesions_per_image[0], self.config.lesions_per_image[1] + 1)
        for _ in range(int(n_lesions)):
            cx, cy = self.rng.uniform(-0.6, 0.6, size=2)
            radius = self.rng.uniform(0.08, 0.22)
            strength = self.rng.uniform(*self.config.susceptibility_range)
            blob = np.exp(-((self.radius_template - radius) ** 2) / (2 * (0.06**2)))
            shift_x = int(cx * self.config.image_size / 2)
            shift_y = int(cy * self.config.image_size / 2)
            blob = np.roll(blob, shift=(shift_x, shift_y), axis=(0, 1))
            susceptibility += strength * blob

        mask = self.radius_template <= 0.95
        susceptibility *= mask
        return susceptibility

    def _forward_field(self, susceptibility: np.ndarray) -> np.ndarray:
        fft = np.fft.fft2(susceptibility)
        field = np.fft.ifft2(fft * self.dipole_kernel).real.astype(np.float32)
        noise = self.rng.normal(0, self.config.noise_std, size=field.shape).astype(np.float32)
        return field + noise

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        target = self._generate_susceptibility()
        field = self._forward_field(target)

        field_tensor = torch.from_numpy(field).unsqueeze(0)
        target_tensor = torch.from_numpy(target).unsqueeze(0)

        return {"field": field_tensor, "susceptibility": target_tensor}
