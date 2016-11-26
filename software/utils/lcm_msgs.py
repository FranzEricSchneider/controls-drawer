import time

import lcmtypes


KNOWN_MESSAGES = {
	'REQUEST_IMAGE' : lcmtypes.image_request_t,
	'IMAGE_RAW' : lcmtypes.image_t
}

def auto_decode(channel, data):
    return KNOWN_MESSAGES[channel].decode(data)

def auto_instantiate(channel):
    return KNOWN_MESSAGES[channel]()

def utime_now():
	return long(time.time() * 1e6)
