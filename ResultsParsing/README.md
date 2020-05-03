# Output parsing
## Event tracing
The event tracing outputs a binary file

The event_trace.ksy file describes the format of the binary file.

Running `kaitai-struct-compiled -t python event_trace.ksy` generates the event_trace.py file. This file is used in parse_events.py to read the metadata and buffers from the file. This requires the kaitai struct python package to be installed as a runtime. See http://doc.kaitai.io/lang_python.html for installation instructions.
