import os
import cv2
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_video", type=str, default="data/valid.mp4")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--angle", type=float, default=45.0)
    parser.add_argument("--output_path", type=str, default="data/output.png")
    
    args = parser.parse_args()
    
    # get frame number
    cap = cv2.VideoCapture(args.input_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    angle = args.angle
    # for normal: pick frame 0 and frame (frame_num * 7/8)
    # for reverse: pick frame frame_num*1/2 and frame (frame_num * 5/8)
    if not args.reverse:
        main_frame_id = 0
        other_frame_id = int(frame_count * (1 - angle/360))
        # interpolate between 1 to (1-angle/360)
        rate = angle / 360 / 3
        mid_frame_id_1 = int(frame_count * (1 - angle/360 + 2*rate))
        mid_frame_id_2 = int(frame_count * (1 - angle/360 + rate))
    else:
        main_frame_id = int(frame_count / 2)
        other_frame_id = int(frame_count * (1/2 + angle/360))-1
        rate = angle / 360 / 3
        mid_frame_id_1 = int(frame_count * (1/2 + rate))
        mid_frame_id_2 = int(frame_count * (1/2 + 2*rate))
    
    # get these frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, main_frame_id)
    _, main_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, other_frame_id)
    _, other_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_id_1)
    _, mid_frame_1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_id_2)
    _, mid_frame_2 = cap.read()
    
    assert main_frame is not None
    assert other_frame is not None
    assert mid_frame_1 is not None
    assert mid_frame_2 is not None
    
    # close video
    cap.release()
    
    # combine these frames: main, mid, other
    main_frame = cv2.resize(main_frame, (640, 360))
    mid_frame_1 = cv2.resize(mid_frame_1, (640, 360))
    mid_frame_2 = cv2.resize(mid_frame_2, (640, 360))
    other_frame = cv2.resize(other_frame, (640, 360))
    
    combined_image = cv2.hconcat([main_frame, mid_frame_1, mid_frame_2, other_frame])

    cv2.imwrite(args.output_path, combined_image)