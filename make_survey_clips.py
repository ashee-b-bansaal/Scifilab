import argparse
import threading
from bisect import bisect_left
import cv2
import os 


def get_frame_size(video_path: str):
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    return (int(width), int(height))


def save_clip(
        video_path: str,
        start_index: int,
        end_index: int,
        output_filename: str,
        FPS: float = 30.0
):
    """
    save a clip based on the specifed start and end index of the specified video 
    
    """
    frame_size = get_frame_size(video_path)
    print(frame_size)
    cap = cv2.VideoCapture(video_path)
    four_cc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_filename,
        four_cc,
        FPS,
        frame_size
    )

    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if index > end_index:
            break
        if index < start_index:
            index += 1
            continue
        video_writer.write(frame)
        index += 1
        
    video_writer.release()
    cap.release()


def event_timestamp_to_frame_indices(
        frame_timestamp: list[float],
        event_timestamp: float,
        before_event_duration: float,
        after_event_duration: float):
    start_index = bisect_left(frame_timestamp, event_timestamp - before_event_duration)
    end_index = bisect_left(frame_timestamp, event_timestamp + after_event_duration)
    return (start_index, end_index)
    
    
def make_clip_frame_indices(
        frame_timestamp: list[float],
        event_timestamp: list[float],
        before_event_duration: list[float],
        after_event_duration: list[float]):
    res = []
    for i in range(len(event_timestamp)):
        res.append(
            event_timestamp_to_frame_indices(
                frame_timestamp,
                event_timestamp[i],
                before_event_duration[i],
                after_event_duration[i]
            ))
    return res


def save_survey_clips(
        video_path: str,
        frame_timestamp: list[float],
        event_timestamp: list[float],
        before_event_duration: list[float],
        after_event_duration: list[float],
        clip_names: list[str],
        output_folder: str
        ):
    assert len(before_event_duration) == len(after_event_duration) and len(before_event_duration) == len(event_timestamp)
    
    frame_indices = make_clip_frame_indices(
        frame_timestamp,
        event_timestamp,
        before_event_duration,
        after_event_duration
    )
    start_indices, end_indices = zip(*frame_indices)
    video_filename_list = [os.path.join(output_folder, clip_names[i]) for i in range(len(event_timestamp))]

    write_video_thread_list = []
    for i in range(len(event_timestamp)):
        write_video_t = threading.Thread(
            target=save_clip,
            args=(video_path,
                  start_indices[i],
                  end_indices[i],
                  video_filename_list[i])
        )
        write_video_thread_list.append(write_video_t)
        write_video_t.start()

    for i in range(len(write_video_thread_list)):
        write_video_thread_list[i].join()

def parse_frame_timestamp_file(file_path) -> list[float]:
    with open(file_path, 'r') as f:
        frame_timestamp = [float(line.rstrip()) for line in f]

    return frame_timestamp

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-vp",
                        "--video-path",
                        help = "path to video file",
                        required=True)
    parser.add_argument("-fp",
                        "--frame-path",
                        help = "path to video frame timestamp file",
                        required=True)
    parser.add_argument("-cp",
                        "--clip-path",
                        help = "output folder for clips")
    # parser.add_argument("-ep",
                        # "--event-path",
                        # help = "path to event log file",
                        # ) # will become required later

    args = parser.parse_args()
    event_timestamp = [1742157642.4331865, 1742157690.8314416 ]
    before_event_duration = [1.0, 5.0]
    after_event_duration = [5.0, 1.0]
    frame_timestamp = parse_frame_timestamp_file(args.frame_path)
    
    # save_survey_clips(
    #     args.video_path,
    #     frame_timestamp,
    #     event_timestamp,
    #     before_event_duration,
    #     after_event_duration,
    #     args.clip_path
    # )

                    
    
    

    
    
