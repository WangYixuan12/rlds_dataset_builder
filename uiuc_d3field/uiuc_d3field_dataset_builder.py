from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
import cv2
from tqdm import tqdm


class UiucD3field(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.1.2')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.1.0': 'Add all data.',
      '1.1.1': 'Downsample to 1fps.',
      '1.1.2': 'Remove some bad data'
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
            'train': self._generate_examples(path='/media/yixuan_2T/dynamic_repr/NeuRay/data'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            print('Parsing episode: ', episode_path)
            is_exp = episode_path.split('/')[6] == 'real_plan'
            # load raw data --> this should change for your dataset
            if is_exp:
                total_steps = min(len(os.listdir(os.path.join(episode_path, 'camera_0', 'color'))), 1000)
            else:
                total_steps = min(len(glob.glob(os.path.join(episode_path, 'camera_0', 'color_*.png'))), 1000)
            # total_steps = 100
            episode = []
            for i in range(0, total_steps, 10):
                curr_state = np.loadtxt(os.path.join(episode_path, 'pose', 'robotiq_base', f'{i}.txt')).astype(np.float32)
                last_state = np.loadtxt(os.path.join(episode_path, 'pose', 'robotiq_base', f'{i-1}.txt')).astype(np.float32) if i > 0 else curr_state
                if is_exp:
                    color_name = f'color/{i}.png'
                    depth_name = f'depth/{i}.png'
                else:
                    color_name = f'color_{i}.png'
                    depth_name = f'depth_{i}.png'
                try:
                    episode.append({
                        'observation': {
                            'image_1': cv2.imread(os.path.join(episode_path, 'camera_0', color_name))[..., ::-1],
                            'depth_1': cv2.imread(os.path.join(episode_path, 'camera_0', depth_name), cv2.IMREAD_ANYDEPTH)[..., np.newaxis],
                            'image_2': cv2.imread(os.path.join(episode_path, 'camera_1', color_name))[..., ::-1],
                            'depth_2': cv2.imread(os.path.join(episode_path, 'camera_1', depth_name), cv2.IMREAD_ANYDEPTH)[..., np.newaxis],
                            'image_3': cv2.imread(os.path.join(episode_path, 'camera_2', color_name))[..., ::-1],
                            'depth_3': cv2.imread(os.path.join(episode_path, 'camera_2', depth_name), cv2.IMREAD_ANYDEPTH)[..., np.newaxis],
                            'image_4': cv2.imread(os.path.join(episode_path, 'camera_3', color_name))[..., ::-1],
                            'depth_4': cv2.imread(os.path.join(episode_path, 'camera_3', depth_name), cv2.IMREAD_ANYDEPTH)[..., np.newaxis],
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
                except:
                    print('Error in ', episode_path)
                    print(os.path.join(episode_path, 'camera_0', color_name))
                    print(os.path.join(episode_path, 'camera_0', depth_name))
                    # print(os.path.join(episode_path, 'camera_1', color_name))
                    # print(os.path.join(episode_path, 'camera_1', depth_name))
                    # print(os.path.join(episode_path, 'camera_2', color_name))
                    # print(os.path.join(episode_path, 'camera_2', depth_name))
                    # print(os.path.join(episode_path, 'camera_3', color_name))
                    # print(os.path.join(episode_path, 'camera_3', depth_name))
                    print(os.path.join(episode_path, 'pose', 'robotiq_base', f'{i}.txt'))
                    print(os.path.join(episode_path, 'pose', 'robotiq_base', f'{i-1}.txt'))
                    break

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        paths = []
        
        exp_path = os.path.join(path, 'real_plan')
        exclude_scenes = ['blank']
        for scene_path in os.listdir(exp_path):
            curr_scene_path = os.path.join(exp_path, scene_path)
            if scene_path in exclude_scenes:
                continue
            for episode_path in os.listdir(curr_scene_path):
                if episode_path[-4:] == '.zip':
                    continue
                if not os.path.exists(os.path.join(curr_scene_path, episode_path, 'pose')):
                    continue
                paths.append(os.path.join(curr_scene_path, episode_path))
        
        demo_path = os.path.join(path, 'dyn_data')
        
        exclude_date = ['extrinsics_cali']
        exclude_scenes = ['wrong_gripper']
        for date_path in os.listdir(demo_path):
            if date_path in exclude_date:
                continue
            curr_date_path = os.path.join(demo_path, date_path)
            for scene_path in os.listdir(curr_date_path):
                curr_scene_path = os.path.join(curr_date_path, scene_path)
                if scene_path in exclude_scenes:
                    continue
                for episode_path in os.listdir(curr_scene_path):
                    if episode_path[-4:] == '.zip':
                        continue
                    if not os.path.exists(os.path.join(curr_scene_path, episode_path, 'pose')):
                        continue
                    paths.append(os.path.join(curr_scene_path, episode_path))
        
        print(f'Found {len(paths)} episodes')
        print(paths)
        for p in tqdm(paths):
            yield _parse_example(p)

        # # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create([paths])
        #         | beam.Map(_parse_example)
        # )

