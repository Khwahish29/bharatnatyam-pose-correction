import cv2
import json
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector
from mmengine.registry import init_default_scope
from mmengine.config import Config

class BharatanatyamPoseCorrection:
    def __init__(self):
        self.setup_models()
        self.load_reference_poses()
        self.setup_gui()
        
    def create_temp_config(self):
        """Create a temporary config file with all required variables"""
        config_content = """
_base_ = ['mmpose::_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=270, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# optimizer
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.0))

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=1024,
        out_channels=133,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'CocoWholeBodyDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = []
val_pipeline = []
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict()
val_dataloader = dict()
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/coco_wholebody_val_v1.0.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
    ))

# evaluators
val_evaluator = dict()
test_evaluator = val_evaluator
"""
        config_path = 'temp_rtmpose_config.py'
        with open(config_path, 'w') as f:
            f.write(config_content)
        return config_path

    def setup_models(self):
        """Initialize pose estimation models"""
        # Detector configuration
        det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
        det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        
        # Create temporary config file for pose estimator
        pose_config = self.create_temp_config()
        pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth'
        
        try:
            # Initialize models
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.detector = init_detector(det_config, det_checkpoint, device=self.device)
            self.pose_estimator = init_model(pose_config, pose_checkpoint, device=self.device)
        finally:
            # Clean up temporary config file
            if Path(pose_config).exists():
                Path(pose_config).unlink()

    def load_reference_poses(self):
        """Load reference pose keypoints from JSON files"""
        self.reference_poses = {}
        reference_dir = Path('reference_poses')
        if not reference_dir.exists():
            print(f"Warning: {reference_dir} does not exist. Creating directory.")
            reference_dir.mkdir(exist_ok=True)
            print("Please run create_reference_poses.py first to generate reference pose data.")
        
        for pose_file in reference_dir.glob('*.json'):
            pose_name = pose_file.stem
            with open(pose_file) as f:
                self.reference_poses[pose_name] = json.load(f)
        
        if not self.reference_poses:
            print("No reference poses found. Adding dummy pose for testing.")
            self.reference_poses["test_pose"] = {"keypoints": [[0, 0, 0]] * 133}

    def setup_gui(self):
        """Setup the GUI interface with support for multiple users"""
        self.root = tk.Tk()
        self.root.title("Bharatanatyam Pose Correction - Multiple Users")
        
        # Frame for user controls
        self.users_frame = ttk.Frame(self.root)
        self.users_frame.pack(pady=10)
        
        # Dictionary to store user selections
        self.user_selections = {}
        
        # Add user button
        add_user_btn = ttk.Button(self.root, text="Add User", command=self.add_user)
        add_user_btn.pack(pady=5)
        
        # Video feed
        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)
        
        # Feedback text (now using a dictionary for multiple users)
        self.feedback_frame = ttk.Frame(self.root)
        self.feedback_frame.pack(pady=10)
        self.feedback_texts = {}
        
        # Start button
        start_button = ttk.Button(self.root, text="Start Pose Correction", command=self.start_correction)
        start_button.pack(pady=10)
        
        # Add initial user
        self.add_user()

    def add_user(self):
        """Add a new user control set"""
        user_id = len(self.user_selections)
        user_frame = ttk.LabelFrame(self.users_frame, text=f"User {user_id + 1}")
        user_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Pose selection for this user
        pose_var = tk.StringVar()
        pose_names = list(self.reference_poses.keys())
        pose_label = ttk.Label(user_frame, text="Select Pose:")
        pose_label.pack()
        pose_dropdown = ttk.Combobox(user_frame, textvariable=pose_var, values=pose_names)
        pose_dropdown.pack(pady=5)
        
        # Store user selections
        self.user_selections[user_id] = {
            'pose_var': pose_var,
            'frame': user_frame
        }
        
        # Add feedback text for this user
        feedback_frame = ttk.LabelFrame(self.feedback_frame, text=f"Feedback - User {user_id + 1}")
        feedback_frame.pack(side=tk.LEFT, padx=10, pady=5)
        feedback_text = tk.Text(feedback_frame, height=5, width=40)
        feedback_text.pack(pady=5)
        self.feedback_texts[user_id] = feedback_text

    def get_pose_keypoints(self, frame):
        """Extract pose keypoints from a frame for multiple people"""
        try:
            # Detect person
            init_default_scope('mmdet')
            det_result = inference_detector(self.detector, frame)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)]
            bboxes = bboxes[:, :4]

            if len(bboxes) == 0:
                print("No people detected in frame")
                return [], []

            # Estimate pose
            init_default_scope('mmpose')
            pose_results = inference_topdown(self.pose_estimator, frame, bboxes)
            data_samples = merge_data_samples(pose_results)

            if hasattr(data_samples, 'pred_instances'):
                keypoints = data_samples.pred_instances.keypoints
                scores = data_samples.pred_instances.keypoint_scores
                return keypoints, scores
            
            print("No pose keypoints detected")
            return [], []
            
        except Exception as e:
            print(f"Error in get_pose_keypoints: {str(e)}")
            return [], []

    def normalize_keypoints(self, kps):
        """Enhanced normalization considering scale, rotation and translation invariance"""
        # Check if keypoints have confidence scores
        has_confidence = kps.shape[1] > 2
        
        # Center the pose using the mid-hip point
        hip_center = (kps[11, :2] + kps[12, :2]) / 2
        normalized_kps = kps.copy()  # Create a copy to avoid modifying the original
        
        # Translate to origin
        normalized_kps[:, :2] = normalized_kps[:, :2] - hip_center
        
        # Scale by torso size (distance between hip center and neck)
        neck = (kps[5, :2] + kps[6, :2]) / 2
        scale = np.linalg.norm(neck - hip_center)
        if scale > 0:
            normalized_kps[:, :2] = normalized_kps[:, :2] / scale
        
        # Normalize rotation using the angle between vertical and spine
        spine_vector = neck - hip_center
        if spine_vector[0] != 0:  # Avoid division by zero
            angle = np.arctan2(spine_vector[1], spine_vector[0])
            rotation_matrix = np.array([
                [np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle), np.cos(-angle)]
            ])
            
            # Apply rotation to all keypoints
            for i in range(len(normalized_kps)):
                # Only rotate keypoints with good confidence if confidence scores are available
                if not has_confidence or (has_confidence and normalized_kps[i, 2] > 0.3):
                    normalized_kps[i, :2] = rotation_matrix @ normalized_kps[i, :2]
        
        return normalized_kps

    def compare_poses(self, current_keypoints, reference_keypoints):
        """Compare poses with detailed Bharatanatyam-specific feedback"""
        feedback = []
        all_parts_correct = True
        
        # Normalize keypoints
        current_normalized = self.normalize_keypoints(current_keypoints)
        reference_normalized = self.normalize_keypoints(reference_keypoints)
        
        # Define Bharatanatyam-specific body parts with their keypoint indices and thresholds
        bharatanatyam_parts = {
            'mudras': {
                'left_hand': list(range(91, 111)),  # Left hand keypoints
                'right_hand': list(range(112, 132)),  # Right hand keypoints
                'threshold': 0.2,
                'conf_threshold': 0.3,
                'importance': 'critical'
            },
            'araimandi': {
                'knees': [13, 14],  # Left and right knees
                'ankles': [15, 16],  # Left and right ankles
                'threshold': 0.15,
                'conf_threshold': 0.5,
                'importance': 'critical'
            },
            'arms': {
                'elbows': [7, 8],  # Left and right elbows
                'shoulders': [5, 6],  # Left and right shoulders
                'threshold': 0.15,
                'conf_threshold': 0.5,
                'importance': 'high'
            },
            'torso': {
                'spine': [5, 6, 11, 12],  # Shoulders and hips
                'threshold': 0.15,
                'conf_threshold': 0.5,
                'importance': 'high'
            },
            'feet': {
                'positions': [15, 16],  # Ankles/feet positions
                'threshold': 0.15,
                'conf_threshold': 0.4,
                'importance': 'medium'
            }
        }

        # Dictionary to store feedback by importance
        feedback_by_importance = {
            'critical': [],
            'high': [],
            'medium': []
        }

        # Check confidence scores
        scores = current_keypoints[:, 2] if current_keypoints.shape[1] > 2 else np.ones(len(current_keypoints))

        # Analyze Mudras (Hand Positions)
        def analyze_mudras():
            for side in ['left_hand', 'right_hand']:
                indices = bharatanatyam_parts['mudras'][side]
                avg_conf = np.mean(scores[indices])
                
                if avg_conf < bharatanatyam_parts['mudras']['conf_threshold']:
                    feedback_by_importance['critical'].append(
                        f"⚠️ {side.replace('_', ' ').title()}: Mudra not clearly visible"
                    )
                    return
                
                hand_diff = np.mean(np.abs(
                    current_normalized[indices, :2] - reference_normalized[indices, :2]
                ))
                
                if hand_diff > bharatanatyam_parts['mudras']['threshold']:
                    direction = self.get_direction_feedback(
                        current_normalized[indices], 
                        reference_normalized[indices]
                    )
                    feedback_by_importance['critical'].append(
                        f"✗ {side.replace('_', ' ').title()}: Adjust mudra position {direction}"
                    )
                else:
                    feedback_by_importance['critical'].append(
                        f"✓ {side.replace('_', ' ').title()}: Correct mudra position"
                    )

        # Analyze Araimandi Position
        def analyze_araimandi():
            knees = bharatanatyam_parts['araimandi']['knees']
            ankles = bharatanatyam_parts['araimandi']['ankles']
            
            # Calculate knee bend angle
            left_knee_angle = self.calculate_angle(
                current_normalized[11],  # Left hip
                current_normalized[13],  # Left knee
                current_normalized[15]   # Left ankle
            )
            right_knee_angle = self.calculate_angle(
                current_normalized[12],  # Right hip
                current_normalized[14],  # Right knee
                current_normalized[16]   # Right ankle
            )
            
            # Ideal Araimandi angle is approximately 120 degrees
            ideal_angle = 120
            angle_threshold = 15
            
            if abs(left_knee_angle - ideal_angle) > angle_threshold or \
               abs(right_knee_angle - ideal_angle) > angle_threshold:
                feedback_by_importance['critical'].append(
                    "✗ Araimandi: Adjust knee bend - aim for half-sitting position"
                )
                if left_knee_angle > ideal_angle + angle_threshold:
                    feedback_by_importance['critical'].append("  - Bend left knee more")
                elif left_knee_angle < ideal_angle - angle_threshold:
                    feedback_by_importance['critical'].append("  - Straighten left knee slightly")
                if right_knee_angle > ideal_angle + angle_threshold:
                    feedback_by_importance['critical'].append("  - Bend right knee more")
                elif right_knee_angle < ideal_angle - angle_threshold:
                    feedback_by_importance['critical'].append("  - Straighten right knee slightly")
            else:
                feedback_by_importance['critical'].append("✓ Araimandi: Good knee position")

        # Analyze Arms Position
        def analyze_arms():
            try:
                # Define arm points for left and right sides
                arm_points = [
                    {"side": "Left", "points": [5, 7, 9]},    # Left shoulder, elbow, wrist
                    {"side": "Right", "points": [6, 8, 10]}   # Right shoulder, elbow, wrist
                ]
                
                for arm in arm_points:
                    shoulder_idx = arm["points"][0]
                    elbow_idx = arm["points"][1]
                    wrist_idx = arm["points"][2]
                    
                    # Check confidence scores for arm points
                    if (scores[shoulder_idx] > 0.3 and 
                        scores[elbow_idx] > 0.3 and 
                        scores[wrist_idx] > 0.3):
                        
                        # Calculate arm angle
                        arm_angle = self.calculate_angle(
                            current_normalized[shoulder_idx],
                            current_normalized[elbow_idx],
                            current_normalized[wrist_idx]
                        )
                        
                        # Compare with reference angle
                        ref_arm_angle = self.calculate_angle(
                            reference_normalized[shoulder_idx],
                            reference_normalized[elbow_idx],
                            reference_normalized[wrist_idx]
                        )
                        
                        if abs(arm_angle - ref_arm_angle) > 20:  # 20 degrees threshold
                            direction = "bend" if arm_angle > ref_arm_angle else "straighten"
                            feedback_by_importance['high'].append(
                                f"✗ {arm['side']} Arm: {direction} your elbow slightly"
                            )
                        else:
                            feedback_by_importance['high'].append(
                                f"✓ {arm['side']} Arm: Good position"
                            )
                    else:
                        feedback_by_importance['high'].append(
                            f"⚠️ {arm['side']} Arm: Not clearly visible"
                        )
                    
            except Exception as e:
                print(f"Error in analyze_arms: {str(e)}")
                feedback_by_importance['high'].append("⚠️ Unable to analyze arms properly")

        # Analyze Torso Alignment
        def analyze_torso():
            spine_points = bharatanatyam_parts['torso']['spine']
            
            # Calculate spine angle relative to vertical
            hip_center = (current_normalized[11, :2] + current_normalized[12, :2]) / 2
            shoulder_center = (current_normalized[5, :2] + current_normalized[6, :2]) / 2
            spine_angle = np.degrees(np.arctan2(
                shoulder_center[0] - hip_center[0],
                shoulder_center[1] - hip_center[1]
            ))
            
            if abs(spine_angle) > 10:  # 10 degrees threshold
                direction = "right" if spine_angle > 0 else "left"
                feedback_by_importance['high'].append(
                    f"✗ Torso: Align your spine - leaning to the {direction}"
                )
            else:
                feedback_by_importance['high'].append(
                    "✓ Torso: Good alignment"
                )

        # Run all analyses
        analyze_mudras()
        analyze_araimandi()
        analyze_arms()
        analyze_torso()

        # Combine feedback in order of importance
        feedback.append("\nCritical Adjustments:")
        feedback.extend(feedback_by_importance['critical'])
        
        feedback.append("\nImportant Adjustments:")
        feedback.extend(feedback_by_importance['high'])
        
        feedback.append("\nMinor Adjustments:")
        feedback.extend(feedback_by_importance['medium'])
        
        return feedback

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def get_direction_feedback(self, current_points, reference_points):
        """Get directional feedback based on average displacement"""
        diff = np.mean(current_points[:, :2] - reference_points[:, :2], axis=0)
        directions = []
        
        if abs(diff[1]) > 0.1:
            directions.append("higher" if diff[1] < 0 else "lower")
        if abs(diff[0]) > 0.1:
            directions.append("right" if diff[0] < 0 else "left")
        
        return f"({', '.join(directions)})" if directions else ""

    def start_correction(self):
        """Start pose correction with improved error handling and user mapping"""
        try:
            # Validate that at least one user has selected a pose
            if not self.user_selections:
                print("Please add at least one user!")
                return
            
            # Validate all users have selected poses
            for user_id, selection in self.user_selections.items():
                if not selection['pose_var'].get():
                    print(f"Please select a pose for User {user_id + 1}")
                    return
            
            # Load reference poses and images for all users
            reference_data = {}
            for user_id, selection in self.user_selections.items():
                selected_pose = selection['pose_var'].get()
                reference_dir = Path('/home/user/mmpose/Bharatnatyam_combined/ideal_pose')
                reference_image_path = reference_dir / f"{selected_pose}.jpg"
                
                if not reference_image_path.exists():
                    print(f"Error: Reference image not found for User {user_id + 1}")
                    return
                
                reference_data[user_id] = {
                    'pose_name': selected_pose,
                    'image': cv2.imread(str(reference_image_path)),
                    'keypoints': np.array(self.reference_poses[selected_pose]['keypoints'])
                }

            # Process video
            video_path = "/home/user/mmpose/Bharatnatyam_combined/user_video/IMG-20241026-WA0062_IMG-20241026-WA0060_2.mp4"
            if not Path(video_path).exists():
                print(f"Error: Video file not found at {video_path}")
                return

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video file")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video")
                    break

                # Create a composite overlay
                overlay = frame.copy()
                
                # Get all person detections and their poses
                keypoints_list, scores_list = self.get_pose_keypoints(frame)
                
                if len(keypoints_list) == 0:
                    # If no people detected, update all users with "No detection"
                    for user_id in self.user_selections:
                        self.feedback_texts[user_id].delete(1.0, tk.END)
                        self.feedback_texts[user_id].insert(tk.END, f"User {user_id + 1}: No detection")
                    self.update_display(frame)
                    self.root.after(30)
                    continue

                # Build a list of detections with computed center x-coordinate for ordering
                detected_persons = []
                for idx, (kpts, scrs) in enumerate(zip(keypoints_list, scores_list)):
                    # Get only the valid keypoints (with confidence > 0.3)
                    if kpts.shape[1] > 2:
                        valid_points = kpts[kpts[:, 2] > 0.3][:, :2]
                    else:
                        valid_points = kpts
                    center_x = np.mean(valid_points[:, 0]) if valid_points.shape[0] > 0 else 0
                    detected_persons.append({'center_x': center_x, 'keypoints': kpts, 'scores': scrs})
                
                # Sort detected persons from leftmost to rightmost (increasing x-coordinate)
                detected_persons.sort(key=lambda person: person['center_x'])
                
                # Map detections to users based on the order users were added
                sorted_user_ids = sorted(self.user_selections.keys())
                for idx, user_id in enumerate(sorted_user_ids):
                    if idx >= len(detected_persons):
                        # Update feedback if no detection for this user
                        self.feedback_texts[user_id].delete(1.0, tk.END)
                        self.feedback_texts[user_id].insert(tk.END, f"User {user_id + 1}: No detection")
                        continue

                    detection = detected_persons[idx]
                    kpts = detection['keypoints']
                    scrs = detection['scores']
                    ref_data = reference_data[user_id]
                    
                    # Scale and align reference overlay for this person
                    scaled_ref = self.scale_reference_to_person(
                        ref_data['image'], 
                        kpts, 
                        frame.shape
                    )
                    
                    overlay = cv2.addWeighted(overlay, 0.8, scaled_ref, 0.2, 0)
                    
                    # Compare poses and update feedback for the corresponding user
                    feedback = self.compare_poses(kpts, ref_data['keypoints'])
                    self.feedback_texts[user_id].delete(1.0, tk.END)
                    self.feedback_texts[user_id].insert(tk.END, f"User {user_id + 1}:\n" + "\n".join(feedback))
                    
                    # Draw skeleton for this detection
                    self.draw_skeleton(overlay, kpts, scrs)

                # Update display with composite overlay
                self.update_display(overlay)
                
                # Add a small delay
                self.root.after(30)

        except Exception as e:
            print(f"Error in start_correction: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if 'cap' in locals():
                cap.release()

    def update_display(self, frame):
        """Helper method to update the display"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.root.update()
        except Exception as e:
            print(f"Error updating display: {str(e)}")

    def scale_reference_to_person(self, reference_image, person_keypoints, frame_shape):
        """Scale and align reference image to match detected person's size and position"""
        try:
            # Check keypoints shape and handle different formats
            if person_keypoints.ndim != 2:
                print(f"Warning: Unexpected keypoints shape: {person_keypoints.shape}")
                return cv2.resize(reference_image, (frame_shape[1], frame_shape[0]))
            
            # Get valid points based on available dimensions
            if person_keypoints.shape[1] > 2:
                # If we have confidence scores (3rd dimension)
                valid_points = person_keypoints[person_keypoints[:, 2] > 0.3][:, :2]
            else:
                # If we only have x,y coordinates
                valid_points = person_keypoints

            if len(valid_points) < 2:
                print("Warning: Not enough valid keypoints detected")
                return cv2.resize(reference_image, (frame_shape[1], frame_shape[0]))
            
            # Calculate bounding box
            x_min, y_min = np.min(valid_points, axis=0)
            x_max, y_max = np.max(valid_points, axis=0)
            
            # Add padding to bounding box (10% on each side)
            width = x_max - x_min
            height = y_max - y_min
            padding_x = width * 0.1
            padding_y = height * 0.1
            
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(frame_shape[1], x_max + padding_x)
            y_max = min(frame_shape[0], y_max + padding_y)
            
            # Calculate scale factor based on height
            person_height = y_max - y_min
            ref_height = reference_image.shape[0]
            scale_factor = person_height / ref_height
            
            # Ensure reasonable scale factor
            scale_factor = min(max(scale_factor, 0.1), 5.0)  # Limit between 0.1x and 5x
            
            # Scale reference image
            new_width = int(reference_image.shape[1] * scale_factor)
            new_height = int(reference_image.shape[0] * scale_factor)
            
            # Ensure minimum size
            new_width = max(new_width, 50)
            new_height = max(new_height, 50)
            
            scaled_ref = cv2.resize(reference_image, (new_width, new_height))
            
            # Create canvas of frame size
            canvas = np.zeros(frame_shape, dtype=np.uint8)
            
            # Calculate position to place scaled reference
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            
            # Calculate placement coordinates
            x_start = max(0, x_center - new_width // 2)
            y_start = max(0, y_center - new_height // 2)
            x_end = min(frame_shape[1], x_start + new_width)
            y_end = min(frame_shape[0], y_start + new_height)
            
            # Ensure valid regions for placement
            if x_end > x_start and y_end > y_start:
                # Calculate valid regions for both source and destination
                dest_width = x_end - x_start
                dest_height = y_end - y_start
                src_width = min(dest_width, scaled_ref.shape[1])
                src_height = min(dest_height, scaled_ref.shape[0])
                
                # Place scaled reference on canvas
                canvas[y_start:y_start+src_height, x_start:x_start+src_width] = \
                    scaled_ref[:src_height, :src_width]
            
            return canvas
        
        except Exception as e:
            print(f"Error in scale_reference_to_person: {str(e)}")
            # Return resized reference image as fallback
            return cv2.resize(reference_image, (frame_shape[1], frame_shape[0]))

    def draw_skeleton(self, frame, keypoints, scores):
        """Draw skeleton with accurate connections for Bharatanatyam poses"""
        # Define skeleton connections
        skeleton = [
            # Body
            (0, 1), (0, 2),  # Nose to eyes
            (1, 3), (2, 4),  # Eyes to ears
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 6), (5, 11), (6, 12),  # Torso
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
            
            # Left Hand (connect wrist to fingers)
            (9, 91),  # Left wrist to left hand
            (91, 92), (92, 93), (93, 94),  # Thumb
            (91, 95), (95, 96), (96, 97), (97, 98),  # Index
            (91, 99), (99, 100), (100, 101), (101, 102),  # Middle
            (91, 103), (103, 104), (104, 105), (105, 106),  # Ring
            (91, 107), (107, 108), (108, 109), (109, 110),  # Pinky
            
            # Right Hand (connect wrist to fingers)
            (10, 112),  # Right wrist to right hand
            (112, 113), (113, 114), (114, 115),  # Thumb
            (112, 116), (116, 117), (117, 118), (118, 119),  # Index
            (112, 120), (120, 121), (121, 122), (122, 123),  # Middle
            (112, 124), (124, 125), (125, 126), (126, 127),  # Ring
            (112, 128), (128, 129), (129, 130), (130, 131),  # Pinky
        ]

        # Draw connections
        for start_idx, end_idx in skeleton:
            if scores[start_idx] > 0.3 and scores[end_idx] > 0.3:
                start_pt = tuple(map(int, keypoints[start_idx, :2]))
                end_pt = tuple(map(int, keypoints[end_idx, :2]))
                
                # Thicker lines for body, thinner for hands
                thickness = 2 if start_idx < 17 else 1
                cv2.line(frame, start_pt, end_pt, (0, 255, 0), thickness)

        # Draw keypoints
        for i, (kp, score) in enumerate(zip(keypoints, scores)):
            if score > 0.3:
                x, y = int(kp[0]), int(kp[1])
                
                # Different colors for different body parts
                if i < 17:  # Body
                    color = (0, 255, 0)  # Green
                    radius = 4
                elif i < 91:  # Face
                    color = (255, 0, 0)  # Blue
                    radius = 2
                else:  # Hands
                    color = (0, 0, 255)  # Red
                    radius = 3
                
                cv2.circle(frame, (x, y), radius, color, -1)

    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = BharatanatyamPoseCorrection()
    app.run()