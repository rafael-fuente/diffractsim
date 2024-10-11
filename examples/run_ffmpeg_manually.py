import ffmpeg

framerate = 20
start_frame = 1
path = "./animation/"
filename = "frame_"
filetype = ".png"  # Can save as jpg or png

# frame2_000070.png

(
    ffmpeg  # pip install ffmpeg-python // Put ffmpeg.exe in same folder (or maybe in a path variable)
    .input(path + filename + '%06d' + filetype, start_number = start_frame, framerate = framerate)  # Note: pattern_type='glob' and "*" wildcard only works in linux
    .output(path + 'animation.mp4')
    .run(overwrite_output = True)
)

# (
#     ffmpeg  # pip install ffmpeg-python // Put ffmpeg.exe in same folder (or maybe in a path variable)
#     .input('./animation/frame_%04d.png', framerate=24)  # Note: pattern_type='glob' and "*" wildcard only works in linux
#     .output('./animation/animation.mp4')
#     .run(overwrite_output=True)
# )
