# Compass: Co-Exploration of Mapping and Hardware for Heterogeneous Multi-Chiplet Accelerators Targeting LLM Inference Service Workloads

---

## Dependencies

This project requires C++17 or higher, g++-10, libssl-dev, and Python 3.11 or higher.

```
sudo apt install g++-10 libssl-dev
```

To install the necessary Python dependencies, run:

```
python3 -m pip install zigzag-dse
```

You also need to install gpytorch for BO search:

```
python3 -m pip install gpytorch
```

---

## Preparation

### Launch Zigzag

You need to open the project in a new screen and go to the `zigzag_call` directory:

```
cd zigzag_call
```

Launch zigzag_call as a local web service:

```
python3 zigzag_call.py
```

### Build the project

Go to another screen and compile the project with:

```
make
```

This will generate an executable named `compass` in the build directory. The executable can be run independently for mapping search or execution on a given hardware configuration:

```
./build/compass <search/exec_config.json> <hardware.json> <search/exec_res.csv>
```

The hardware JSON may set `tensor_parall` to select the number of output-channel shards for eligible merged-model OutProj and FFN GEMMs. When the value is greater than one, each shard is exposed as an independent mapping node and may be placed on a different chiplet. Residual, normalization, and activation nodes retain the original dependencies and execute after their producer shards.

---

## Quick Try

Create a new directory:

```
mkdir try
cd try
```

Copy the example configuration files for Compass and hardware:

```
cp ../config/search_config_example_gpt_merged.json ./
cp ../config/exec_config_example_gpt_merged.json ./
cp ../config/hardware_ws.json ./
```

Run Compass to search for a mapping on the Simba architecture:

```
../build/compass search_config_example_gpt_merged.json hardware_ws.json search_res.csv
```

Afterward, in the try folder, you will find:

`best_mapping.json`: the best mapping solution found by Compass

`search_process.csv`: the mapping search process

`search_res.csv`: performance metrics of the best solution on the dataset used during the search

Then, run execution using the obtained mapping:

```
../build/compass exec_config_example_gpt_merged.json hardware_ws.json exec_res.csv
```

You will then find the following in the try folder:

`exec_res.csv`: performance metrics on the test dataset using the provided mapping

`exec_latency_detail.json, exec_energy_detail.json, exec_mc_detail.json`: detailed latency, energy, and monetary cost from the final execution

---

## Run Exp

To perform hardware sampling under a given compute budget and workload using `bo.py` (which internally calls compass):

```
python3 bo.py <exp_dir_path> <workload [prefill/decode]> <scale [64/512/2048]>
```

To directly run the comparative experiments from the paper:

```
python3 exp.py
```
