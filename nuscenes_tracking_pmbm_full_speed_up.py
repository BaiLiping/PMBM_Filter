import json
import numpy as np
from datetime import datetime
from pyquaternion import Quaternion
from typing import List, Dict
from tqdm import tqdm
import copy
from filter_config import FilterConfig, get_gaussian_density_NuScenes_CV
from pmbm import PoissonMultiBernoulliMixture
from object_detection import ObjectDetection

from nuscenes.nuscenes import NuScenes


def format_result(sample_token: str,
                  translation: List[float],
                  size: List[float],
                  yaw: float,
                  velocity: List[float],
                  tracking_id: int,
                  tracking_name: str,
                  tracking_score: float) -> Dict:
    """
    Format tracking result for 1 single target as following
    sample_result {
        "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
        "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
        "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
        "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
        "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
        "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
        "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                           Note that the tracking_name cannot change throughout a track.
        "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                           We average over frame level scores to compute the track level score.
                                           The score is used to determine positive and negative tracks via thresholding.
    }
    """
    sample_result = {}
    sample_result['sample_token'] = sample_token
    sample_result['translation'] = translation
    sample_result['size'] = size
    sample_result['rotation'] = Quaternion(angle=yaw, axis=[0, 0, 1]).elements.tolist()
    sample_result['velocity'] = velocity
    sample_result['tracking_id'] = tracking_id
    sample_result['tracking_name'] = tracking_name
    sample_result['tracking_score'] = tracking_score
    return sample_result


def main():
    # load NuScenes
    data_root = '/home/zhubinglab/Desktop/mmdetection3d/data/nuscenes'
    version = 'v1.0-test'
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    with open('/home/zhubinglab/Desktop/mmdetection3d/data/nuscenes/official_inference_result/centerpoint_test_detection.json', 'r') as f:
        data_all_original = json.load(f)
    data_all=data_all_original['results']

    # load scene token
    #with open('test_scene_tokens_centerPoint.json', 'r') as infile:
    #    val_scene_tokens = json.load(infile)

    nuscenes_data = NuScenes(version = version, dataroot=data_root, verbose=False)
    scenes=nuscenes_data.scene
    frames=nuscenes_data.sample



    # init tracking results for the whole val set
    tracking_results = {}

    #for _, scene_token in tqdm(val_scene_tokens.items()):
    for scene in scenes:
        frames_for_this_scene = []
        for frame in frames:
            if frame['scene_token']==scene['token']:
                frames_for_this_scene.append(frame)
                # set the frames in corret order
                # notice that NuScenes database does not provide the numerical ordering of the frames
                # it provides previous and next frame token information
                unordered_frames = copy.deepcopy(frames_for_this_scene)
                ordered_frames=[]
                # looping until the unordered frame is an empty set
                while len(unordered_frames)!=0:
                    for current_frame in unordered_frames:
                        # if it is the first frame
                        if current_frame['prev']=='':
                            ordered_frames.append(current_frame)
                            # set current token
                            current_frame_token_of_current_scene = current_frame['token']
                            unordered_frames.remove(current_frame)
                
                    # find the next frame of current frame
                    if current_frame['prev']==current_frame_token_of_current_scene:
                        ordered_frames.append(current_frame)
                        # reset current frame
                        current_frame_token_of_current_scene=current_frame['token']
                        unordered_frames.remove(current_frame)

        for frame_idx, frame in enumerate(ordered_frames):

            # initialize filter
            config = FilterConfig(state_dim=6, measurement_dim=3)
            density = get_gaussian_density_NuScenes_CV()
            pmbm_filter = PoissonMultiBernoulliMixture(config, density)


            tracking_results[frame['token']] = []
            # invoke filter and extract estimation
            measurements = data_all[frame['token']]
            if len(measurements) > 0:
                estimation = pmbm_filter.run(measurements)
                # log estimation
                for target_id, target_est in estimation.items():
                    sample_result = format_result(frame['token'],
                                                  target_est['translation'] + [target_est['height']],
                                                  target_est['size'],
                                                  target_est['orientation'],
                                                  target_est['velocity'],
                                                  target_id,
                                                  target_est['class'],
                                                  target_est['score'])
                    tracking_results[frame['token']].append(sample_result)

    # save tracking result
    meta = {'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False}
    output_data = {'meta': meta, 'results': tracking_results}
    with open('/home/zhubinglab/Desktop/QianSubmission-{}.json'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), 'w') as outfile:
        json.dump(output_data, outfile)


if __name__ == '__main__':
    main()