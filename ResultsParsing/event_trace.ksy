meta:
  id: event_trace
  endian: le
seq:
  - id: meta
    type: event
    repeat: eos

  
types:
  event:
    seq:
      - id: source_lp
        type: u4
      - id: destination_lp
        type: u4
      - id: virtual_send_time
        type: f4
      - id: virtual_recv_time
        type: f4
      - id: real_time
        type: f4
      - id: model_data_size
        type: u4
      - id: buf
        size: model_data_size
