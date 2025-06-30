#!/bin/bash

phase=$1
../../build/compass ./compass_config_exec.json ./${phase}_phase/best_${phase}_hardware.json ./${phase}_phase/exec_out/exec_res_${phase}.csv > ./${phase}_phase/exec_out/exec_out_${phase}.log