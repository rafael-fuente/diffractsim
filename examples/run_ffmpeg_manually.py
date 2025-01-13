import ffmpeg

framerate = 25
start_frame = 1
path = "./animation/"
filename = "frame_"
filetype = ".png"  # Can save as jpg or png

# pip install ffmpeg-python // Add ffmpeg.exe to system path variable (Or just put in same folder)
(
    ffmpeg
    .input(path + filename + '%06d' + filetype, start_number = start_frame, framerate = framerate)  # Note: pattern_type='glob' and "*" wildcard only work in linux
    .output(path + 'animation.mp4', vcodec='png', movflags='faststart')  # crf=1, **{'qscale:v': 1}
    .run(overwrite_output = True)
    # for Android use: vcodec='libvpx-vp9', pix_fmt='yuv420p', video_bitrate=200000000, movflags='faststart'
)
