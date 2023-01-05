from pathlib import Path
from typing import Union, Tuple, List, Optional

import h5py
import numpy as np


class DataSplitter:
    """Splits movie into train/test sets of frames"""
    def __init__(
        self,
        movie_path: Union[str, Path],
        seed: Optional[int] = None
    ):
        """

        Parameters
        ----------
        movie_path
            Path to movie
        seed
            Seed for reproducibility
        """
        self._movie_path = Path(movie_path)
        self._seed = seed

    def get_train_val_split(
            self,
            train_frac: float = 0.7,
            window_size: int = 30
    ) -> Tuple[List[int], List[int]]:
        """
        Returns train/val split for the movie. Each split contains the center
        frame indices.

        Parameters
        ----------
        train_frac
            The fraction of center frames to reserve for the training set
        window_size
            Number of frames before and after a center frame

        Returns
        -------
        Tuple of train, val center frame indices
        """
        with h5py.File(self._movie_path, 'r') as f:
            nframes = f['data'].shape[0]

        # min, max center frame must be window_size away from beginning/end
        # since the frame needs to have window_size before/after it
        all_frames = np.arange(window_size,
                               nframes - window_size)

        # Reserving the last (1 - train_frac) frames for validation
        n_train = int(len(all_frames) * train_frac)

        train = all_frames[:n_train]
        val = all_frames[n_train:]

        rng = np.random.default_rng(self._seed)
        rng.shuffle(train)

        return train, val
