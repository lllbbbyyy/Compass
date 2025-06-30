
# 全局可选项定义
chiplet_count_options = [1, 2, 4, 8, 16, 24, 36, 48, 64]
chiplet_type_list = ["NVDLA", "Eyeriss"]
buffer_size_list = [512, 1024, 2048, 4096, 8192, 16384]  # KB
compute_unit_list = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
nop_bw_options = [32, 64, 128, 256]
dram_bw_options = [16, 32, 64, 128, 256]
decode_micro_batch_options = [1, 2, 4, 8, 16, 32, 64, 128]
prefill_micro_batch_options = [1, 2, 4]
mixed_micro_batch_options = [1, 2, 3, 6, 11, 22, 33, 66]

# chiplet_count_options = [1] + list(range(2,128+1,2))
# chiplet_type_list = ["NVDLA", "Eyeriss"]
# buffer_size_list = list(range(512,65537,512))
# compute_unit_list = list(range(512,65537,512))
# nop_bw_options = list(range(16,256,16))
# dram_bw_options = list(range(16,256,16))
# decode_micro_batch_options = [1, 2, 4, 8, 16, 32, 64, 128]
# prefill_micro_batch_options = [1, 2, 4]
# mixed_micro_batch_options = [1, 2, 3, 6, 11, 22, 33, 66]