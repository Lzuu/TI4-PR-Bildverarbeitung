"""Render invariants; these test pixels and geometry, not a particular artwork."""
import cv2
import numpy as np

from suit_rig import alpha_over, segment_transform, SuitRenderer, load_assets
from test_pose_tracking import sample_pose


def test_alpha_over_uses_premultiplied_color_without_a_black_fringe():
    image=np.full((40,40,3),100,np.uint8)
    sprite=np.zeros((10,10,4),np.uint8)
    sprite[2:8,2:8]=(240,220,200,128)
    alpha_over(image,sprite,np.array([[1,0,10],[0,1,10]],np.float32))
    assert np.allclose(image[14,14],(170,160,150),atol=1)
    assert np.array_equal(image[10,10],(100,100,100))


def test_segment_affine_maps_both_anchors_and_rotates_width_perpendicular():
    anchors=((50,10),(50,90))
    matrix=segment_transform(anchors,(20,30),(100,30),width=24,source_width=60)
    mapped=cv2.transform(np.array([anchors],np.float32),matrix)[0]
    assert np.allclose(mapped,((20,30),(100,30)),atol=1e-4)
    assert abs(matrix[0,0])<1e-5


def test_left_and_right_gloves_are_mirrored_without_moving_wrist_anchors():
    renderer=SuitRenderer()
    endpoints=np.float32([[50,30],[50,70]])
    left=renderer._part('glove','glove_l',endpoints,30)
    right=renderer._part('glove','glove_r',endpoints,30)
    assert np.linalg.det(left['matrix'][:,:2])*np.linalg.det(right['matrix'][:,:2])<0
    h,w=renderer.assets['glove'].image.shape[:2]
    anchors=np.asarray(renderer.meta['glove']['anchors'],np.float32)*(w,h)
    for part in (left,right):
        assert np.allclose(cv2.transform(anchors[None],part['matrix'])[0],endpoints,atol=1e-4)


def test_partially_offscreen_sprite_does_not_wrap_or_change_far_background():
    frame=np.full((50,50,3),30,np.uint8)
    sprite=np.full((20,20,4),255,np.uint8)
    alpha_over(frame,sprite,np.array([[1,0,-15],[0,1,8]],np.float32))
    assert (frame[10,1]==255).all()
    assert (frame[:,20:]==30).all()


def test_compositing_respects_current_person_mask_including_holes():
    frame=np.full((200,160,3),35,np.uint8)
    raw=np.full_like(frame,(100,140,180))
    mask=np.zeros((200,160),np.uint8)
    mask[20:190,30:130]=255
    mask[100:115,75:85]=0
    renderer=SuitRenderer()
    out=renderer.render(frame,raw,mask,pose=sample_pose(),face=(64,18,32,36),now=0)
    assert np.array_equal(out[105,80],(35,35,35))
    assert np.array_equal(out[180,150],(35,35,35))


def test_original_face_is_visible_even_when_input_canvas_head_is_already_painted():
    raw=np.full((240,320,3),(75,130,190),np.uint8)
    cv2.circle(raw,(150,60),4,(20,25,30),-1)
    cv2.circle(raw,(170,60),4,(20,25,30),-1)
    frame=np.full_like(raw,240)
    mask=np.full(raw.shape[:2],255,np.uint8)
    out=SuitRenderer().render(frame,raw,mask,face=(130,30,60,80),now=0)
    # Facial features and the centre of the face survive the clear glass.
    assert np.max(np.abs(out[60,150].astype(int)-raw[60,150]))<=10
    assert np.mean(np.abs(out[45:95,140:180].astype(float)-raw[45:95,140:180]))<10


def test_current_marker_is_restored_from_raw_even_if_yolo_excluded_it():
    raw=np.full((200,160,3),90,np.uint8)
    keep=np.zeros((200,160),np.uint8);keep[80:100,75:85]=255
    raw[keep>0]=(180,0,255)
    mask=np.full((200,160),255,np.uint8);mask[keep>0]=0
    out=SuitRenderer().render(np.zeros_like(raw),raw,mask,pose=sample_pose(),keep=keep,now=0)
    assert np.array_equal(out[keep>0],raw[keep>0])


def test_camera_backdrop_does_not_leak_into_the_helmet_cavity():
    raw=np.full((240,320,3),(0,200,0),np.uint8)
    mask=np.zeros(raw.shape[:2],np.uint8)
    cv2.ellipse(mask,(160,70),(30,44),0,0,360,255,-1)
    cv2.rectangle(mask,(125,105),(195,239),255,-1)
    raw[mask>0]=(75,130,190)
    out=SuitRenderer().render(np.full_like(raw,(25,20,15)),raw,mask,
                              face=(130,30,60,80),now=0)
    # Beside the face but inside the large opening: dark helmet interior, no green wall.
    assert int(out[70,124,1])-max(int(out[70,124,0]),int(out[70,124,2]))<15
    assert np.max(np.abs(out[65,160].astype(int)-raw[65,160]))<=10


def test_missing_player_is_a_noop_and_clears_head_history():
    renderer=SuitRenderer()
    frame=np.full((200,160,3),40,np.uint8)
    mask=np.full((200,160),255,np.uint8)
    renderer.render(frame.copy(),frame,mask,face=(60,15,40,45),now=0)
    out=renderer.render(frame.copy(),frame,np.zeros_like(mask),now=.05)
    assert np.array_equal(out,frame)
    assert renderer.head is None


def test_all_foreground_mask_does_not_overflow_distance_shading():
    frame=np.full((80,80,3),80,np.uint8)
    with np.errstate(over='raise'):
        SuitRenderer().render(frame.copy(),frame,np.full((80,80),255,np.uint8),now=0)


def test_packaged_helmet_has_real_transparent_opening_and_parts_have_alpha():
    assets,_=load_assets()
    for sprite in assets.values():
        assert sprite.image[0,0,3]==0
        assert np.count_nonzero(sprite.image[:,:,3])>100
    helmet=assets['helmet']
    h,w=helmet.image.shape[:2]
    assert helmet.image[h//2,w//2,3]==0
    assert helmet.opening[h//2,w//2]>0


def test_expired_pose_and_face_restore_original_head_instead_of_painting_it_white():
    renderer=SuitRenderer()
    raw=np.full((200,160,3),(70,125,180),np.uint8)
    mask=np.full((200,160),255,np.uint8)
    renderer.render(raw.copy(),raw,mask,face=(60,15,40,45),now=0)
    out=renderer.render(raw.copy(),raw,mask,now=.4)
    assert np.array_equal(out[20:50,65:95],raw[20:50,65:95])


def test_head_size_stays_calibrated_when_yunet_briefly_loses_the_face():
    renderer=SuitRenderer()
    pose=sample_pose();pose.points[2,:2]=(70,30);pose.points[5,:2]=(90,30)
    raw=np.full((200,160,3),90,np.uint8);mask=np.full((200,160),255,np.uint8)
    renderer.render(raw.copy(),raw,mask,pose=pose,face=(60,15,40,45),now=0)
    moved=pose.copy();moved.points[:,:2]+=(3,1)
    renderer.render(raw.copy(),raw,mask,pose=moved,face=None,now=.04)
    assert abs(renderer.head[0][2]-40)<2
    assert abs(renderer.head[0][3]-45)<2


def test_renderer_does_not_modify_the_original_camera_buffer():
    raw=np.full((200,160,3),(60,110,160),np.uint8);before=raw.copy()
    SuitRenderer().render(raw.copy(),raw,np.full((200,160),255,np.uint8),pose=sample_pose(),now=0)
    assert np.array_equal(raw,before)


def test_tracked_body_without_reliable_head_never_paints_the_face_white():
    pose=sample_pose();pose.confidence[:11]=0
    raw=np.full((200,160,3),(70,125,180),np.uint8)
    mask=np.full((200,160),255,np.uint8)
    out=SuitRenderer().render(raw.copy(),raw,mask,pose=pose,face=None,now=0)
    assert np.array_equal(out[15:35,65:95],raw[15:35,65:95])


def test_head_expires_after_short_dropout_and_a_new_source_resets_it():
    renderer=SuitRenderer()
    frame=np.full((240,320,3),90,np.uint8)
    mask=np.full(frame.shape[:2],255,np.uint8)
    renderer.render(frame.copy(),frame,mask,face=(130,30,60,80),now=0,source=0)
    renderer.render(frame.copy(),frame,mask,now=.05,source=0)
    assert renderer.head is not None
    renderer.render(frame.copy(),frame,mask,now=.5,source=0)
    assert renderer.head is None
    renderer.render(frame.copy(),frame,mask,face=(130,30,60,80),now=.6,source=0)
    renderer.render(frame.copy(),frame,mask,now=.63,source=1)
    assert renderer.head is None


def test_torso_and_raised_forearm_are_distinct_transformed_parts():
    renderer=SuitRenderer()
    pose=sample_pose();pose.points[13,:2]=(35,30);pose.points[15,:2]=(55,5)
    raw=np.full((200,160,3),80,np.uint8)
    mask=np.full((200,160),255,np.uint8)
    renderer.render(raw.copy(),raw,mask,pose=pose,now=0)
    names=[p['name'] for p in renderer.last_parts]
    assert 'torso' in names and 'forearm_l' in names
    arm=next(p for p in renderer.last_parts if p['name']=='forearm_l')
    assert np.allclose(arm['endpoints'],pose.points[[13,15],:2])


def test_torso_waist_adapts_to_body_outline_not_only_distance_between_hip_joints():
    pose=sample_pose()
    raw=np.full((200,160,3),80,np.uint8)
    narrow=np.zeros((200,160),np.uint8);narrow[10:190,35:125]=255
    wide=np.zeros_like(narrow);wide[10:190,10:150]=255
    r=SuitRenderer();r.render(raw.copy(),raw,narrow,pose=pose,now=0)
    first=next(p for p in r.last_parts if p['name']=='torso')['endpoints']
    r=SuitRenderer();r.render(raw.copy(),raw,wide,pose=pose,now=0)
    second=next(p for p in r.last_parts if p['name']=='torso')['endpoints']
    assert np.linalg.norm(second[3]-second[2])>np.linalg.norm(first[3]-first[2])
