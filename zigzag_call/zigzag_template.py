workload_template='''
- id: 0
  operator_type: Gemm
  equation: O[m][n]+=I[m][k]*W[k][n]
  dimension_relations: []
  loop_dims: [M, K, N]
  loop_sizes: [{m}, {k}, {n}]
  operand_precision:
    W: 8
    I: 8
    O: 8
    O_final: 8
'''


hardware_os_template='''
name: os_chiplet

memories:

  output_registers:
    size: 8  # 8 word-bits * 64 cluster_size
    r_cost: 0.009 # added to mac energy
    w_cost: 0.009 # added to mac energy
    area: 0
    latency: 1
    operands: [O]  # Weights
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 8
        bandwidth_max: 8
        allocation: 
          - O, tl
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 8
        bandwidth_max: 8
        allocation: 
          - O, fh
          - O, fl
    served_dimensions: []

  weight_buffer:
    size: 65536
    r_cost: 0.56
    w_cost: 0.67
    area: 0
    latency: 1
    operands: [I2]  # Weights
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - I2, tl
      - name: w_port_1
        type: write
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - I2, fh
    served_dimensions: [D3, D4]

  accumulation_buffer:
    size: 65536
    r_cost: 0.56
    w_cost: 0.67
    area: 0
    latency: 1
    operands: [O]  # Partial sums
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - O, tl
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - O, fl
          - O, fh
    served_dimensions: [D3, D4]

  input_buffer:
    size: 65536 # 8*1024*8
    r_cost: 1.04
    w_cost: 0.67
    area: 0
    latency: 1
    operands: [I1]  # Input activations
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - I1, tl
      - name: w_port_1
        type: write
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - I1, fh
    served_dimensions: [D3, D4]

  global_buffer:
    size: {buffer}  # 2MB
    r_cost: 2.08
    w_cost: 2.25
    area: 0
    latency: 1
    operands: [I1, I2, O]  # Input activations, weights, partial sums
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 1024
        bandwidth_max: 1024
        allocation: 
          - I1, tl
          - I2, tl
          - O, tl
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 1024
        bandwidth_max: 1024
        allocation: 
          - I1, fh
          - I2, fh
          - O, fl
          - O, fh
    served_dimensions: [D1, D2, D3, D4]

  dram:
    size: 10000000000000000
    r_cost: 0
    w_cost: 0
    area: 0
    latency: 0
    operands: [I1, I2, O]
    ports:
      - name: rw_port_1
        type: read_write
        bandwidth_min: 10000000000000000
        bandwidth_max: 10000000000000000
        allocation: 
          - I1, fh
          - I1, tl
          - I2, fh
          - I2, tl
          - O, fh
          - O, tl
          - O, fl
          - O, th
    served_dimensions: [D1, D2, D3, D4]


operational_array:
  unit_energy: 0.009
  unit_area: 1  # unit
  # D1/2 = 4x4 PE array. Each PE has 8 vector MACS (D3) that process 8 elements (D4) in parallel
  dimensions: [D1, D2, D3, D4]
  sizes: {mac}

'''

hardware_ws_template='''
name: ws_chiplet

memories:

  weight_registers:
    size: 8  # 8 word-bits * 64 cluster_size
    r_cost: 0.009 # added to mac energy
    w_cost: 0.009 # added to mac energy
    area: 0
    latency: 1
    operands: [I2]  # Weights
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 8
        bandwidth_max: 8
        allocation: 
          - I2, tl
      - name: w_port_1
        type: write
        bandwidth_min: 8
        bandwidth_max: 8
        allocation: 
          - I2, fh
    served_dimensions: []

  weight_buffer:
    size: 65536 # 8*1024*8
    r_cost: 0.56
    w_cost: 0.67
    area: 0
    latency: 1
    operands: [I2]  # Weights
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - I2, tl
      - name: w_port_1
        type: write
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - I2, fh
    served_dimensions: [D3, D4]

  accumulation_buffer:
    size: 65536 #8 * 1024 *8
    r_cost: 0.56
    w_cost: 0.67
    area: 0
    latency: 1
    operands: [O]  # Partial sums
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 256  # partial sums are 24 bits * 8 units reading in parallel
        bandwidth_max: 256
        allocation: 
          - O, tl
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - O, fl
          - O, fh
    served_dimensions: [D3, D4]

  input_buffer:
    size: 65536 # 8*1024*8
    r_cost: 1.04
    w_cost: 0.67
    area: 0
    latency: 1
    operands: [I1]  # Input activations
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - I1, tl
      - name: w_port_1
        type: write
        bandwidth_min: 256
        bandwidth_max: 256
        allocation: 
          - I1, fh
    served_dimensions: [D3, D4]

  global_buffer:
    size: {buffer}  # 2MB
    r_cost: 2.08
    w_cost: 2.25
    area: 0
    latency: 1
    operands: [I1, I2, O]  # Input activations, weights, partial sums
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 1024
        bandwidth_max: 1024
        allocation: 
          - I1, tl
          - I2, tl
          - O, tl
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 1024
        bandwidth_max: 1024
        allocation: 
          - I1, fh
          - I2, fh
          - O, fl
          - O, fh
    served_dimensions: [D1, D2, D3, D4]

  dram:
    size: 10000000000000000
    r_cost: 0
    w_cost: 0
    area: 0
    latency: 0
    operands: [I1, I2, O]
    ports:
      - name: rw_port_1
        type: read_write
        bandwidth_min: 10000000000000000
        bandwidth_max: 10000000000000000
        allocation: 
          - I1, fh
          - I1, tl
          - I2, fh
          - I2, tl
          - O, fh
          - O, tl
          - O, fl
          - O, th
    served_dimensions: [D1, D2, D3, D4]


operational_array:
  unit_energy: 0.009  # Refine with more accurate data if available
  unit_area: 1  # unit
  # D1/2 = 4x4 PE array. Each PE has 8 vector MACS (D3) that process 8 elements (D4) in parallel
  dimensions: [D1, D2, D3, D4]
  sizes: {mac}

'''

hardware_tesla='''
name: npu_like

memories:
  rf_1B:
    size: 8
    r_cost: 0.01
    w_cost: 0.01
    area: 0
    latency: 1
    operands: [I2]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 8
        bandwidth_max: 8
        allocation: 
          - I2, tl
      - name: w_port_1
        type: write
        bandwidth_min: 8
        bandwidth_max: 8
        allocation: 
          - I2, fh
    served_dimensions: [D2, D3]

  rf_4B:
    size: 32
    r_cost: 0.022
    w_cost: 0.022
    area: 0
    latency: 1
    operands: [O]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, tl
      - name: r_port_2
        type: read
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, fh
      - name: w_port_2
        type: write
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, fl
    served_dimensions: []

  sram_1KB_I:
    size: 8192
    r_cost: 4.78
    w_cost: 5.59
    area: 0
    latency: 1
    operands: [I1]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 64
        bandwidth_max: 256
        allocation: 
          - I1, tl
      - name: w_port_1
        type: write
        bandwidth_min: 64
        bandwidth_max: 256
        allocation: 
          - I1, fh
    served_dimensions: [D1, D2, D3]

  sram_1KB_W:
    size: 8192
    r_cost: 4.78
    w_cost: 5.59
    area: 0
    latency: 1
    operands: [I2]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 64
        bandwidth_max: 256
        allocation: 
          - I2, tl
      - name: w_port_1
        type: write
        bandwidth_min: 64
        bandwidth_max: 256
        allocation: 
          - I2, fh
    served_dimensions: [D1, D2, D3]

  sram_2MB:
    size: {buffer}
    r_cost: 208.08
    w_cost: 189.2
    area: 0
    latency: 1
    operands: [I1, I2, O]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 64
        bandwidth_max: 1024
        allocation: 
          - I1, tl
          - I2, tl
          - O, tl
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 64
        bandwidth_max: 1024
        allocation: 
          - I1, fh
          - I2, fh
          - O, fl
          - O, fh
    served_dimensions: [D1, D2, D3]

  dram:
    size: 10000000000000000
    r_cost: 0
    w_cost: 0
    area: 0
    latency: 0
    operands: [I1, I2, O]
    ports:
      - name: rw_port_1
        type: read_write
        bandwidth_min: 10000000000000000
        bandwidth_max: 10000000000000000
        allocation: 
          - I1, fh
          - I1, tl
          - I2, fh
          - I2, tl
          - O, fh
          - O, tl
          - O, fl
          - O, th
    served_dimensions: [D1, D2, D3]

operational_array:
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1, D2, D3]
  sizes: {mac}

'''

hardware_tpu='''
name: tpu_like

memories:
  rf_128B:
    size: 1024
    r_cost: 0.095
    w_cost: 0.095
    area: 0
    latency: 1
    operands: [I2]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 8
        bandwidth_max: 8
        allocation: 
          - I2, tl
      - name: w_port_1
        type: write
        bandwidth_min: 8
        bandwidth_max: 8
        allocation: 
          - I2, fh
    served_dimensions: [] # Fully unrolled over all multipliers

  rf_2B:
    size: 16
    r_cost: 0.021
    w_cost: 0.021
    area: 0
    latency: 1
    operands: [O]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, tl
      - name: r_port_2
        type: read
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, fh
      - name: w_port_2
        type: write
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, fl
    served_dimensions: []

  sram_2MB:
    size: {buffer}
    r_cost: 208.16
    w_cost: 189.4
    area: 0
    latency: 1
    operands: [I1, O]
    ports:
      - fh: w_port_1
        tl: r_port_1
      - fh: w_port_1
        tl: r_port_1
        fl: w_port_1
        th: r_port_1
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 64
        bandwidth_max: 2048
        allocation: 
          - I1, tl
          - O, tl
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 64
        bandwidth_max: 2048
        allocation: 
          - I1, fh
          - O, fh
          - O, fl
    served_dimensions: [D1, D2]

  dram:
    size: 10000000000000000
    r_cost: 0
    w_cost: 0
    area: 0
    latency: 0
    operands: [I1, I2, O]
    ports:
      - name: rw_port_1
        type: read_write
        bandwidth_min: 10000000000000000
        bandwidth_max: 10000000000000000
        allocation: 
          - I1, fh
          - I1, tl
          - I2, fh
          - I2, tl
          - O, fh
          - O, tl
          - O, fl
          - O, th
    served_dimensions: [D1, D2]

operational_array:
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1, D2]
  sizes: {mac}

'''

hardware_ascend='''
name: ascend_like

memories:
  rf_1B:
    size: 8
    r_cost: 0.01
    w_cost: 0.01
    area: 0
    latency: 1
    operands: [I2]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 8
        bandwidth_max: 8
        allocation: 
          - I2, tl
      - name: w_port_1
        type: write
        bandwidth_min: 8
        bandwidth_max: 8
        allocation: 
          - I2, fh
    served_dimensions: [D3, D4]

  rf_2B:
    size: 16
    r_cost: 0.02
    w_cost: 0.02
    area: 0
    latency: 1
    operands: [O]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, tl
      - name: r_port_2
        type: read
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, fh
      - name: w_port_2
        type: write
        bandwidth_min: 16
        bandwidth_max: 16
        allocation: 
          - O, fl
    served_dimensions: [D2]

  rf_64KB_I:
    size: 65536
    r_cost: 26.56
    w_cost: 30.72
    area: 0
    latency: 1
    operands: [I1]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 64
        bandwidth_max: 512
        allocation: 
          - I1, tl
      - name: w_port_1
        type: write
        bandwidth_min: 64
        bandwidth_max: 512
        allocation: 
          - I1, fh
    served_dimensions: [D1, D2, D3, D4]

  rf_64KB_W:
    size: 65536
    r_cost: 50.16
    w_cost: 108.0
    area: 0
    latency: 1
    operands: [I2]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 64
        bandwidth_max: 2048
        allocation: 
          - I2, tl
      - name: w_port_1
        type: write
        bandwidth_min: 64
        bandwidth_max: 2048
        allocation: 
          - I2, fh
    served_dimensions: [D1, D2, D3, D4]

  sram_256KB_O:
    size: 2097152
    r_cost: 123.2
    w_cost: 212.8
    area: 0
    latency: 1
    operands: [O]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 64
        bandwidth_max: 2048
        allocation: 
          - O, tl
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 64
        bandwidth_max: 2048
        allocation: 
          - O, fh
          - O, fl
    served_dimensions: [D1, D2, D3, D4]

  sram_2MB:
    size: {buffer}
    r_cost: 208.08
    w_cost: 189.2
    area: 0
    latency: 1
    operands: [I1, I2, O]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 64
        bandwidth_max: 1024
        allocation: 
          - I1, tl
          - I2, tl
          - O, tl
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 64
        bandwidth_max: 1024
        allocation: 
          - I1, fh
          - I2, fh
          - O, fl
          - O, fh
    served_dimensions: [D1, D2, D3, D4]

  dram:
    size: 10000000000000000
    r_cost: 0
    w_cost: 0
    area: 0
    latency: 0
    operands: [I1, I2, O]
    ports:
      - name: rw_port_1
        type: read_write
        bandwidth_min: 10000000000000000
        bandwidth_max: 10000000000000000
        allocation: 
          - I1, fh
          - I1, tl
          - I2, fh
          - I2, tl
          - O, fh
          - O, tl
          - O, fl
          - O, th
    served_dimensions: [D1, D2, D3, D4]

operational_array:
  input_precision: [8, 8]
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1, D2, D3, D4]
  sizes: {mac}

'''

mapping_os_template='''
- name: default
  spatial_mapping:
    D1:
      - M, {}
    D2:
      - N, {}
    D3:
      - M, {}
    D4:
      - N, {}
  temporal_ordering:
    - [K, "*"]
    - [N, "*"]
    - [M, "*"]

  memory_operand_links:
    O: O
    W: I2
    I: I1

'''

mapping_ws_template='''
- name: default
  spatial_mapping:
    D1:
      - K, {}
    D2:
      - N, {}
    D3:
      - K, {}
    D4:
      - N, {}
  temporal_ordering:
    - [M, "*"]
    - [N, "*"]
    - [K, "*"]

  memory_operand_links:
    O: O
    W: I2
    I: I1

'''

mapping_tesla='''
- name: default
  spatial_mapping:
    D1:
      - N, {}
    D2:
      - M, {}
    D3:
      - M, {}
  memory_operand_links:
    O: O
    W: I2
    I: I1

'''

mapping_tpu='''
- name: default
  spatial_mapping:
    D1:
      - N, {}
    D2:
      - K, {}
  memory_operand_links:
    O: O
    W: I2
    I: I1

'''


mapping_ascend='''
- name: default
  spatial_mapping:
    D1:
      - N, {}
    D2:
      - K, {}
    D3:
      - M, {}
    D4:
      - M, {}
  memory_operand_links:
    O: O
    W: I2
    I: I1

'''

hardware_template={
    'os': hardware_os_template,
    'ws': hardware_ws_template,
    'tesla': hardware_tesla,
    'tpu': hardware_tpu,
    'ascend': hardware_ascend
}

mapping_template={
    'os': mapping_os_template,
    'ws': mapping_ws_template,
    'tesla': mapping_tesla,
    'tpu': mapping_tpu,
    'ascend': mapping_ascend
}
