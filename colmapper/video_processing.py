import glob
import os
import subprocess
import shutil
import concurrent.futures
import multiprocessing
from tqdm import tqdm


def get_video_frame_count(video_path):
    """
    Get the total number of frames in a video file using ffprobe.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        tqdm.write(f"Warning: Could not get frame count for {video_path}: {str(e)}")
        return 0


def get_video_resolution(video_path):
    """
    Get the resolution (widthxheight) of a video file using ffprobe.
    """
    # AI-generated comment: This function executes ffprobe to retrieve the width and height of a video stream.
    # It is designed to handle potential errors gracefully if ffprobe fails or returns unexpected output.
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split("x"))
        return width, height
    except (subprocess.CalledProcessError, ValueError) as e:
        tqdm.write(f"Warning: Could not get resolution for {video_path}: {str(e)}")
        return None, None


def process_single_video(video_info, progress_bar):
    """
    Process a single video file. This function is designed to be called by ThreadPoolExecutor.

    Args:
        video_info: Tuple containing (video_path, data_dir, camera_dir_name, downsample_factor)
        progress_bar: tqdm progress bar to update
    """
    video_path, data_dir, camera_dir_name, downsample_factor = video_info
    video_filename = os.path.basename(video_path)

    # AI-generated comment: Use the provided sequential camera name (e.g., 'cam01')
    # for the output directory instead of the video's filename.
    camera_output_base_dir = os.path.join(data_dir, camera_dir_name)
    output_frames_dir = os.path.join(camera_output_base_dir, "images")

    # Get total frames for this video
    total_frames = get_video_frame_count(video_path)

    # AI-generated comment: Get original video resolution to calculate downscaled resolution.
    original_width, original_height = get_video_resolution(video_path)

    # AI-generated comment: Check if frames already exist. If so, skip processing to avoid re-extraction.
    # This check is based on the presence of .png files in the target directory.
    if os.path.exists(output_frames_dir):
        existing_frames = glob.glob(os.path.join(output_frames_dir, "*.png"))
        if existing_frames:
            frames_to_update = (
                total_frames if total_frames > 0 else len(existing_frames)
            )
            progress_bar.update(frames_to_update)
            return True

    # Create the camera-specific base directory and the 'images' subdirectory
    os.makedirs(output_frames_dir, exist_ok=True)

    if total_frames == 0:
        tqdm.write(
            f"Warning: Could not determine frame count for {video_filename}, progress may be inaccurate"
        )

    # AI-generated comment: Base FFmpeg command for frame extraction.
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        video_path,
        "-loglevel",
        "error",
        "-qscale:v",
        "2",
        "-start_number",
        "0",
        "-threads",
        "0",  # Let ffmpeg use all available threads for this video
        os.path.join(output_frames_dir, "%04d.png"),
    ]

    # AI-generated comment: If resolution was found, downscale the video to half its original resolution.
    if original_width and original_height:
        # AI-generated comment: Calculate new resolution (half of original) and insert scaling parameters into the ffmpeg command.
        new_width = original_width // downsample_factor
        new_height = original_height // downsample_factor
        ffmpeg_command.insert(5, f"scale={new_width}:{new_height}")
        ffmpeg_command.insert(5, "-vf")
    else:
        tqdm.write(
            f"Warning: Could not get resolution for {video_filename}. Processing at original resolution."
        )

    try:
        # Start ffmpeg process
        process = subprocess.Popen(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        # Monitor the output directory for new frames
        last_frame = 0
        while process.poll() is None:
            try:
                # Count current frames in output directory
                current_frames = len(
                    [f for f in os.listdir(output_frames_dir) if f.endswith(".png")]
                )
                if current_frames > last_frame:
                    # Update progress bar with the number of new frames
                    progress_bar.update(current_frames - last_frame)
                    last_frame = current_frames
            except Exception as e:
                tqdm.write(
                    f"Warning: Error monitoring progress for {video_filename}: {str(e)}"
                )

            # Small sleep to prevent excessive CPU usage
            import time

            time.sleep(0.1)

        # Check if process completed successfully
        if process.returncode == 0:
            return True
        else:
            stdout, stderr = process.communicate()
            if stderr:
                tqdm.write("FFmpeg stderr:\n" + stderr)
            return False

    except FileNotFoundError:
        tqdm.write(
            "Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH."
        )
        return False


# https://github.com/hustvl/4DGaussians/blob/843d5ac636c37e4b611242287754f3d4ed150144/scripts/llff2colmap.py#L101
def extract_first_frames(data_dir):
    """
    Extracts the first frame of each video in the given directory.
    """
    image_paths = []
    videos = glob.glob(os.path.join(data_dir, "cam[0-9][0-9]"))
    videos = sorted(videos)
    for index, video_path in enumerate(videos):
        image_path = os.path.join(video_path, "images", "0000.png")
        image_paths.append(image_path)
    goal_dir = os.path.join(data_dir, "image_colmap")

    image_name_list = []
    for index, image in enumerate(image_paths):
        image_name = image.split("/")[-1].split(".")
        image_name[0] = "r_%03d" % index
        # breakpoint()
        image_name = ".".join(image_name)
        image_name_list.append(image_name)
        goal_path = os.path.join(goal_dir, image_name)
        shutil.copy(image, goal_path)

    # Moves all .mp4 videos in data_dir into a new directory called "videos"
    videos_dir = os.path.join(data_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    for video in glob.glob(os.path.join(data_dir, "*.mp4")):
        shutil.move(video, os.path.join(videos_dir, os.path.basename(video)))


def extract_frames(data_dir, downsample_factor=1):
    """
    Finds all common video files and processes them in parallel using multiple threads.
    For each video, creates a sequentially named subdirectory 'cam_XX'
    and extracts frames into an 'images' subfolder within it.
    The 'downsample_factor` controls the resolution reduction; e.g., a factor of 2 halves the resolution.
    """
    # Define common video file extensions to search for
    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.wmv"]
    found_video_files = []
    for ext in video_extensions:
        found_video_files.extend(glob.glob(os.path.join(data_dir, ext)))
        found_video_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))

    # Remove duplicates if patterns overlap
    found_video_files = sorted(list(set(found_video_files)))

    if not found_video_files:
        tqdm.write(
            f"No video files ({', '.join(video_extensions)}) found directly in {data_dir}. Nothing to process."
        )
        return

    # Determine number of threads to use
    num_threads = max(1, multiprocessing.cpu_count() - 1)

    # Calculate total frames across all videos
    total_frames = sum(get_video_frame_count(video) for video in found_video_files)

    # Create progress bar for total frames
    progress_bar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")

    # AI-generated comment: Prepare video info for parallel processing.
    # Each video is assigned a sequential camera name (e.g., 'cam01', 'cam02')
    # based on its sorted filename, which will be used for the output directory.
    video_infos = [
        (video_path, data_dir, f"cam{i+1:02d}", downsample_factor)
        for i, video_path in enumerate(found_video_files)
    ]

    # Process videos in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks and get futures
        futures = [
            executor.submit(process_single_video, video_info, progress_bar)
            for video_info in video_infos
        ]

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Get the result (True/False) if needed
            except Exception as e:
                tqdm.write(f"Error in video processing: {str(e)}")

    # Close the progress bar
    progress_bar.close()

    extract_first_frames(data_dir)

