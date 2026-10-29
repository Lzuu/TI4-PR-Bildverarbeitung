"""Pose contracts: coordinates, confidence, source identity and short dropouts."""
import cv2
import numpy as np

from pose_tracking import Pose, PoseTracker, crop_transform, decode_landmarks, detector_anchors, oriented_crop


def sample_pose(dx=0):
    points = np.zeros((33, 3), np.float32)
    points[:, :2] = (80 + dx, 70)
    for index, xy in {0:(80,30), 11:(55,65), 12:(105,65), 13:(40,95),
                      14:(120,95), 15:(35,125), 16:(125,125),
                      23:(65,125), 24:(95,125), 25:(65,155), 26:(95,155),
                      27:(65,185), 28:(95,185)}.items():
        points[index,:2] = (xy[0]+dx,xy[1])
    return Pose(points, np.ones(33,np.float32))


def test_crop_transform_preserves_aspect_and_inverse_at_image_edge():
    matrix = crop_transform((-20, 10, 120, 180), 256)
    points = np.array([[0,0],[99,79],[50,40]],np.float32)
    mapped = cv2.transform(points[None],matrix)[0]
    restored = cv2.transform(mapped[None],cv2.invertAffineTransform(matrix))[0]
    assert np.allclose(restored,points,atol=1e-4)
    assert matrix[0,0] == matrix[1,1]


def test_person_detector_anchor_layout_has_expected_order_and_count():
    anchors=detector_anchors()
    assert anchors.shape==(2254,2)
    assert np.allclose(anchors[0],(.5/28,.5/28))
    assert np.array_equal(anchors[0],anchors[1])
    assert np.allclose(anchors[1568],(.5/14,.5/14))


def test_oriented_crop_centres_hips_and_maps_rotated_up_vector_upwards():
    hips=np.float32([100,150]);up=np.float32([140,70])
    matrix=oriented_crop(hips,up)
    projected=cv2.transform(np.array([[hips,up]],np.float32),matrix)[0]
    assert np.allclose(projected[0],(128,128),atol=1e-4)
    assert abs(projected[1,0]-128)<1e-4
    assert projected[1,1]<128


def test_decoder_uses_visibility_and_presence_and_recovers_original_coordinates():
    raw=np.zeros((39,5),np.float32)
    raw[:,:3]=(128,128,0)
    raw[:,3:]=10
    raw[15,4]=-10
    pose=decode_landmarks(raw.reshape(1,-1),np.array([[2,0,-10],[0,2,-20]],np.float32))
    assert np.allclose(pose.points[0,:2],(69,74))
    assert pose.confidence[15]<.01
    assert pose.confidence[11]>.99


def test_tracking_is_adaptive_and_does_not_strongly_lag_a_fast_move():
    tracker=PoseTracker(None,interval=0)
    first=tracker.stabilize(sample_pose(),0)
    near=tracker.stabilize(sample_pose(1),.03)
    fast=tracker.stabilize(sample_pose(35),.06)
    assert 80<near.points[0,0]<81
    assert fast.points[0,0]>110
    assert first.points[0,0]==80


def test_missing_body_immediately_hides_parts_and_resets_tracking():
    tracker=PoseTracker(lambda frame,mask:sample_pose(),interval=.1)
    frame=np.zeros((200,160,3),np.uint8)
    mask=np.full((200,160),255,np.uint8)
    assert tracker.update(frame,mask,now=0,source=0) is not None
    assert tracker.update(frame,np.zeros_like(mask),now=.03,source=0) is None
    assert tracker.current is None


def test_source_switch_forces_inference_even_with_same_canvas():
    calls=[]
    def infer(frame,mask):
        calls.append(1)
        return sample_pose()
    tracker=PoseTracker(infer,interval=1)
    frame=np.zeros((200,160,3),np.uint8)
    mask=np.full((200,160),255,np.uint8)
    tracker.update(frame,mask,now=0,source=(0,0))
    tracker.update(frame,mask,now=.03,source=(1,0))
    assert len(calls)==2


def test_failed_pose_expires_instead_of_leaving_floating_limbs():
    calls=[sample_pose(),None,None]
    tracker=PoseTracker(lambda f,m:calls.pop(0),interval=0,max_age=.15)
    frame=np.zeros((200,160,3),np.uint8)
    mask=np.full((200,160),255,np.uint8)
    assert tracker.update(frame,mask,now=0) is not None
    assert tracker.update(frame,mask,now=.05) is not None
    assert tracker.update(frame,mask,now=.2) is None


def test_bad_joint_is_not_fabricated_from_a_bbox():
    pose=sample_pose()
    pose.confidence[15]=0
    assert pose.segment(13,15) is None
    assert pose.segment(11,13) is not None
