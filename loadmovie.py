import subprocess as sp
import itertools
import numpy as np

FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS


def load_movie(file, frames=None):
    command = [ FFMPEG_BIN,
                '-i', file,
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-']
    pipe_in = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

    if frames is None:
        frames = itertools.count(start=0, step=1)

    frameidx = -1
    for i in frames:
        while frameidx != i:
            raw_image = pipe_in.stdout.read(1280*720*3)
            frameidx += 1


for i in range(N):
    print('Frame %i' % (i,))
    # read 1280x720*3 bytes (= 1 frame)
    # transform the byte read into a numpy array
    if not raw_image:
        break
    image =  numpy.fromstring(raw_image, dtype='uint8')

    I = image.reshape((720,1280,3))
    # throw away the data in the pipe's buffer.
    pipe_in.stdout.flush()
    pipe.stdin.write( process_img(noiselevel).tostring() )