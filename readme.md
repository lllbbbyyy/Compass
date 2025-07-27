# Compass: Mapping and Hardware Exploration of LLM Inference Workloads on Multi-Chiplet Accelerators

---

## Dependencies

This project requires C++17 or later. To install the necessary Python dependencies, run:
```
pip install -r requirements.txt
```
---

## How to run

First, compile the project with:
```
make
```
This will generate an executable named `compass` in the build directory. The executable can be run independently for mapping search or execution on a given hardware configuration:
```
./build/compass <search/exec_config.json> <hardware.json> <search/exec_res.csv>
```
To perform hardware sampling under a given compute budget and workload using `BO_LNS.py` (which internally calls compass):
```
python3 BO_LNS.py <exp_dir_path> <workload [prefill/decode]> <scale [72/512/2048]>
```
To directly run the comparative experiments from the paper:
```
python3 exp.py
```

---

## Quick Try

Create a new directory:
```
mkdir try
cd try
```
Copy the example configuration files for Compass and Simba hardware:
```
cp ../config/simba_search_config_example.json ./
cp ../config/simba_exec_config_example.json ./
cp ../config/simba_hardware_prefill_config.json ./
```
Run Compass to search for a mapping on the Simba architecture:
```
../build/compass simba_search_config_example.json simba_hardware_prefill_config.json search_res.csv
```
Afterward, in the try folder, you will find:
`best_mapping.json`: the best mapping solution found by Compass
`search_process.csv`: the mapping search process
`search_res.csv`: performance metrics of the best solution on the dataset used during the search

Then, run execution using the obtained mapping:
```
../build/compass simba_exec_config_example.json simba_hardware_prefill_config.json exec_res.csv
```
You will then find the following in the try folder:
`exec_res.csv`: performance metrics on the test dataset using the provided mapping
`exec_latency_detail.json, exec_energy_detail.json, exec_mc_detail.json`: detailed latency, energy, and monetary cost from the final execution
