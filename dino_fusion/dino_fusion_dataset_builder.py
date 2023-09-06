from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
import cv2


class DinoFusion(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_1': tfds.features.Image(
                            shape=(360, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='camera 1 RGB observation.',
                        ),
                        'depth_1': tfds.features.Image(
                            shape=(360, 640, 1),
                            dtype=np.uint16,
                            encoding_format='png',
                            doc='camera 1 depth observation.',
                        ),
                        'image_2': tfds.features.Image(
                            shape=(360, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='camera 2 RGB observation.',
                        ),
                        'depth_2': tfds.features.Image(
                            shape=(360, 640, 1),
                            dtype=np.uint16,
                            encoding_format='png',
                            doc='camera 2 depth observation.',
                        ),
                        'image_3': tfds.features.Image(
                            shape=(360, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='camera 3 RGB observation.',
                        ),
                        'depth_3': tfds.features.Image(
                            shape=(360, 640, 1),
                            dtype=np.uint16,
                            encoding_format='png',
                            doc='camera 3 depth observation.',
                        ),
                        'image_4': tfds.features.Image(
                            shape=(360, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='camera 4 RGB observation.',
                        ),
                        'depth_4': tfds.features.Image(
                            shape=(360, 640, 1),
                            dtype=np.uint16,
                            encoding_format='png',
                            doc='camera 4 depth observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(4,4),
                            dtype=np.float32,
                            doc='Robot end-effector state',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(3,),
                        dtype=np.float32,
                        doc='Robot displacement from last frame',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/media/yixuan_2T/dynamic_repr/NeuRay/data/real_plan/utensils/2023-09-01-16-22-48-305914'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            total_steps = len(os.listdir(os.path.join(episode_path, 'camera_0', 'color')))
            # total_steps = 100
            episode = []
            for i in range(total_steps):
                curr_state = np.loadtxt(os.path.join(episode_path, 'pose', 'robotiq_base', f'{i}.txt')).astype(np.float32)
                last_state = np.loadtxt(os.path.join(episode_path, 'pose', 'robotiq_base', f'{i-1}.txt')).astype(np.float32) if i > 0 else curr_state
                episode.append({
                    'observation': {
                        'image_1': cv2.imread(os.path.join(episode_path, 'camera_0', 'color', f'{i}.png')),
                        'depth_1': cv2.imread(os.path.join(episode_path, 'camera_0', 'depth', f'{i}.png'), cv2.IMREAD_ANYDEPTH)[..., np.newaxis],
                        'image_2': cv2.imread(os.path.join(episode_path, 'camera_1', 'color', f'{i}.png')),
                        'depth_2': cv2.imread(os.path.join(episode_path, 'camera_1', 'depth', f'{i}.png'), cv2.IMREAD_ANYDEPTH)[..., np.newaxis],
                        'image_3': cv2.imread(os.path.join(episode_path, 'camera_2', 'color', f'{i}.png')),
                        'depth_3': cv2.imread(os.path.join(episode_path, 'camera_2', 'depth', f'{i}.png'), cv2.IMREAD_ANYDEPTH)[..., np.newaxis],
                        'image_4': cv2.imread(os.path.join(episode_path, 'camera_3', 'color', f'{i}.png')),
                        'depth_4': cv2.imread(os.path.join(episode_path, 'camera_3', 'depth', f'{i}.png'), cv2.IMREAD_ANYDEPTH)[..., np.newaxis],
                        'state': curr_state,
                    },
                    'action': curr_state[:3, 3] - last_state[:3, 3],
                    'discount': 1.0,
                    'reward': 0.0,
                    'is_first': i == 0,
                    'is_last': i == (total_steps - 1),
                    'is_terminal': i == (total_steps - 1),
                    'language_instruction': '',
                    'language_embedding': np.zeros(512).astype(np.float32),
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        yield _parse_example(path)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        beam = tfds.core.lazy_imports.apache_beam
        return (
                beam.Create([path])
                | beam.Map(_parse_example)
        )

