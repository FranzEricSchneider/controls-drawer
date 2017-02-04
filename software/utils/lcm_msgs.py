import numpy as np
import time

import lcmtypes


KNOWN_MESSAGES = {
    'IMAGE_CROPPED' : lcmtypes.image_t,
    'IMAGE_RAW' : lcmtypes.image_t,
    'POSITION_COMMAND' : lcmtypes.relative_position_t,
    'REQUEST_IMAGE' : lcmtypes.image_request_t,
    'TOOL_STATE' : lcmtypes.tool_state_t,
}

def auto_decode(channel, data):
    return KNOWN_MESSAGES[channel].decode(data)

def auto_instantiate(channel):
    newMsg = KNOWN_MESSAGES[channel]()
    newMsg.utime = utime_now()
    return newMsg

def utime_now():
    return long(time.time() * 1e6)

def nparray_to_image_t_data(frame):
    return frame.flatten()

def image_t_to_nparray(image_t):
    frame = np.array([datum for datum in image_t.data], dtype=np.uint8)
    if image_t.request.format == image_t.request.FORMAT_GRAY:
        frame = frame.reshape(image_t.height, image_t.width)
    elif image_t.request.format == image_t.request.FORMAT_BGR:
        frame = frame.reshape(image_t.height, image_t.width, 3)
    else:
        raise Exception
    return frame

def lcmobj_handle_msg(lcmobj, timeout):
    rfds, wfds, efds = select.select([lcmobj.fileno()], [], [], timeout)
    if rfds:
        lcmobj.handle()
