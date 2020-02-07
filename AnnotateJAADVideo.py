import pickle
import pprint
import cv2
import sys
import shutil
import os
from pathlib import Path
from jaad_data import JAAD

# utils
pp = pprint.PrettyPrinter(indent=2)


def main():
    """
    Arguments:
        argv[1]: video id. e.g. video_0227
        argv[2]: jaad path. e.g. "./"
        argv[3]: new parent folder for output final images. e.g. "./newImg"
    Example:
        python3 AnnotateJAADVideo.py video_0227 . ./newImg
    """
    print("Recieved Arguments: ", str(sys.argv))
    vid = sys.argv[1]
    jaad_path = sys.argv[2]
    newpath_parent = sys.argv[3]
    final_images_path = Path(newpath_parent, vid)

    imdb = JAAD(data_path=jaad_path)

    # pp.pprint([imdb._get_annotations(vid), imdb._get_ped_appearance(vid), imdb._get_ped_attributes(
    #     vid), imdb._get_traffic_attributes(vid), imdb._get_vehicle_attributes(vid)])

    # main annotation data for ped
    video_annotations = imdb._get_annotations(vid)
    # ped_appearance['0_227_1720b'].keys() = ['pose_front', 'pose_back', 'pose_left', 'pose_right', 'clothes_below_knee', 'clothes_upper_light', 'clothes_upper_dark', 'clothes_lower_light', 'clothes_lower_dark', 'backpack', 'bag_hand', 'bag_elbow', 'bag_shoulder', 'bag_left_side', 'bag_right_side', 'cap', 'hood', 'sunglasses', 'umbrella', 'phone', 'baby', 'object', 'stroller_cart', 'bicycle_motorcycle', 'frames']
    ped_appearance = imdb._get_ped_appearance(vid)
    # get ped attributes data (age, crossing, gender, motion_direction, signalized, etc)
    ped_attributes = imdb._get_ped_attributes(vid)
    # get traffic attributes
    traf_attributes = imdb._get_traffic_attributes(vid)
    # get vehicle attributes (speed)
    veh_attributes = imdb._get_vehicle_attributes(vid)

    ped_annotations = video_annotations["ped_annotations"]
    # print(ped_annotations.keys(), len(ped_annotations))
    print("Video width, height, total frames: ", video_annotations["width"], video_annotations["height"], video_annotations["num_frames"])

    # Prepare frame list
    FrameList = []
    for index in range(video_annotations["num_frames"]):
        framepath = imdb._get_image_path(vid, index)
        image = cv2.imread(framepath)
        FrameList.append(image)

    # Get vehicle status (speed) and draw to upper left corner. key=frame number
    for key in veh_attributes:
        val = veh_attributes[key]
        print("Processing Vehicle Attribute for Frame " + str(key))

        image = FrameList[key]

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.72
        # BGR
        color = (0, 0, 255)
        txt_thickness = 2
        string = imdb._map_scalar_to_text("vehicle", val)
        image = cv2.putText(image, string, (10, 40), font, fontScale, color, txt_thickness, cv2.LINE_4, False)
        FrameList[key] = image

    # Draw ped annotations. key=ped id
    for key in ped_annotations:
        if key.endswith("b"):
            ped_data = ped_annotations[key]
            ped_behavior = ped_data["behavior"]
            """ debug
            print('check length1:', len(ped_data['frames']), len(ped_data['bbox']), len(ped_data['occlusion']))
            print('check length2:', len(ped_behavior['cross']), len(ped_behavior['reaction']), len(ped_behavior['hand_gesture']), len(ped_behavior['look']), len(ped_behavior['action']), len(ped_behavior['nod']))
            """
            # loop frames
            for index in range(len(ped_data["frames"])):
                frame_num = ped_data["frames"][index]
                bbox = ped_data["bbox"][index]
                occlusion = ped_data["occlusion"][index]
                cross = ped_behavior["cross"][index]
                reaction = ped_behavior["reaction"][index]
                hand_gesture = ped_behavior["hand_gesture"][index]
                look = ped_behavior["look"][index]
                action = ped_behavior["action"][index]
                nod = ped_behavior["nod"][index]

                age = ped_attributes[key]["age"]
                m_dir = ped_attributes[key]["motion_direction"]

                # draw bbox
                print("Processing Ped Annotation for Frame " + str(frame_num))

                image = FrameList[frame_num]

                start_point = (int(bbox[0]), int(bbox[1]))
                end_point = (int(bbox[2]), int(bbox[3]))
                # Blue color in BGR
                color = (255, 0, 0)
                # Line thickness in px
                thickness = 2
                image = cv2.rectangle(image, start_point, end_point, color, thickness)

                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.66
                color = (0, 0, 255)
                txt_thickness = 1
                text = [
                    "Id: " + key,
                    "Look: " + str(look),
                    "Hand-Gest: " + imdb._map_scalar_to_text("hand_gesture", hand_gesture),
                    "Cross: " + str(cross),
                    "Age: " + imdb._map_scalar_to_text("age", age),
                    "M-Dir: " + imdb._map_scalar_to_text("motion_direction", m_dir),
                ]
                txt_x = int(bbox[2]) + 2
                txt_y = int(bbox[3])
                for t in text:
                    image = cv2.putText(image, t, (txt_x, txt_y), font, fontScale, color, txt_thickness, cv2.LINE_AA, False)
                    txt_y = txt_y - 19
                # cv2.imwrite(newpath, image)
                FrameList[frame_num] = image

    # write new image files
    # create paths if not exist
    final_images_path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(str(final_images_path))
    final_images_path.mkdir(parents=True, exist_ok=True)
    for index in range(video_annotations["num_frames"]):
        oldpath = imdb._get_image_path(vid, index)
        newpath = str(Path(final_images_path, Path(oldpath).name))
        cv2.imwrite(newpath, FrameList[index])

    # Call ffmpeg program to generate video file
    os.system("rm " + str(Path(newpath_parent, "new_" + vid + ".mp4")))
    os.system(
        "ffmpeg -f image2 -framerate 30 -i " + str(Path(final_images_path, "%05d.png")) + " -vcodec libx264 -f mp4 -q:v 3 " + str(Path(newpath_parent, "new_" + vid + ".mp4"))
    )
    # Clean up of temp directories
    shutil.rmtree(str(final_images_path))


if __name__ == "__main__":
    main()

""" Note: How to generate mp4 video from image files
/video_0227$ ffmpeg -f image2 -framerate 30 -i %05d.png -vcodec libx264 -f mp4 -q:v 3 ../new_video_0227.mp4
"""
