import lcmtypes

KNOWN_MESSAGES = {
	'RAW_IMAGE_REQUEST' : lcmtypes.image_request_t,
	'IMAGE_RAW' : lcmtypes.image_t
}

def auto_decode(channel, data):
    return KNOWN_MESSAGES[channel].decode(data)

def auto_instantiate(channel):
    return KNOWN_MESSAGES[channel]()
