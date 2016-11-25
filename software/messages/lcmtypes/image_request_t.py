"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class image_request_t(object):
    __slots__ = ["utime", "n_arguments", "arg_names", "arg_values", "action_id", "name", "dest_channel"]

    def __init__(self):
        self.utime = 0
        self.n_arguments = 0
        self.arg_names = []
        self.arg_values = []
        self.action_id = 0
        self.name = ""
        self.dest_channel = ""

    def encode(self):
        buf = BytesIO()
        buf.write(image_request_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">qb", self.utime, self.n_arguments))
        for i0 in range(self.n_arguments):
            __arg_names_encoded = self.arg_names[i0].encode('utf-8')
            buf.write(struct.pack('>I', len(__arg_names_encoded)+1))
            buf.write(__arg_names_encoded)
            buf.write(b"\0")
        for i0 in range(self.n_arguments):
            __arg_values_encoded = self.arg_values[i0].encode('utf-8')
            buf.write(struct.pack('>I', len(__arg_values_encoded)+1))
            buf.write(__arg_values_encoded)
            buf.write(b"\0")
        buf.write(struct.pack(">q", self.action_id))
        __name_encoded = self.name.encode('utf-8')
        buf.write(struct.pack('>I', len(__name_encoded)+1))
        buf.write(__name_encoded)
        buf.write(b"\0")
        __dest_channel_encoded = self.dest_channel.encode('utf-8')
        buf.write(struct.pack('>I', len(__dest_channel_encoded)+1))
        buf.write(__dest_channel_encoded)
        buf.write(b"\0")

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != image_request_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return image_request_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = image_request_t()
        self.utime, self.n_arguments = struct.unpack(">qb", buf.read(9))
        self.arg_names = []
        for i0 in range(self.n_arguments):
            __arg_names_len = struct.unpack('>I', buf.read(4))[0]
            self.arg_names.append(buf.read(__arg_names_len)[:-1].decode('utf-8', 'replace'))
        self.arg_values = []
        for i0 in range(self.n_arguments):
            __arg_values_len = struct.unpack('>I', buf.read(4))[0]
            self.arg_values.append(buf.read(__arg_values_len)[:-1].decode('utf-8', 'replace'))
        self.action_id = struct.unpack(">q", buf.read(8))[0]
        __name_len = struct.unpack('>I', buf.read(4))[0]
        self.name = buf.read(__name_len)[:-1].decode('utf-8', 'replace')
        __dest_channel_len = struct.unpack('>I', buf.read(4))[0]
        self.dest_channel = buf.read(__dest_channel_len)[:-1].decode('utf-8', 'replace')
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if image_request_t in parents: return 0
        tmphash = (0x75722daee6cb304f) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if image_request_t._packed_fingerprint is None:
            image_request_t._packed_fingerprint = struct.pack(">Q", image_request_t._get_hash_recursive([]))
        return image_request_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)
