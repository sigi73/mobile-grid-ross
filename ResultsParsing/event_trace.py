# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

from pkg_resources import parse_version
from kaitaistruct import __version__ as ks_version, KaitaiStruct, KaitaiStream, BytesIO


if parse_version(ks_version) < parse_version('0.7'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.7 or later is required, but you have %s" % (ks_version))

class EventTrace(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.meta = []
        i = 0
        while not self._io.is_eof():
            self.meta.append(self._root.Event(self._io, self, self._root))
            i += 1


    class Event(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.source_lp = self._io.read_u4le()
            self.destination_lp = self._io.read_u4le()
            self.virtual_send_time = self._io.read_f4le()
            self.virtual_recv_time = self._io.read_f4le()
            self.real_time = self._io.read_f4le()
            self.model_data_size = self._io.read_u4le()
            self.buf = self._io.read_bytes(self.model_data_size)



