import sys
from pathlib import Path
import shutil
from zigzag.api import get_hardware_performance_zigzag

name='ascend'

working_dir = Path(__file__).parent.resolve()

workload = working_dir / 'workload' / 'gemm.yaml'
hardware = working_dir / 'hardware' / f'{name}.yaml'
mapping = working_dir / 'mapping' / f'{name}.yaml'
output_dir = working_dir / 'outputs'

energy, latency, _ =get_hardware_performance_zigzag(workload=str(workload),
                                                            accelerator=str(hardware),
                                                            mapping=str(mapping),
                                                            opt='EDP',
                                                            lpf_limit=5,
                                                            dump_folder=str(output_dir))

print(f'Energy: {energy}, Latency: {latency}')