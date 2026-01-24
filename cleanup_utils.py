import os

def delete_video_and_frames(video_id, video_dir="videos"):
    video_path = os.path.join(video_dir, f"{video_id}.webm")

    if os.path.exists(video_path):
        os.remove(video_path)
        return True
    
    return False
