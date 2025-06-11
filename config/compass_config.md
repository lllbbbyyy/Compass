"run_mode": a dict, must contatin "type", can be one of: "GA" "random" "exec"
for "GA" "random", should be orginzed as:
{
    "type":"GA",
    "best_solution_save_path":"./tmp/best_solution.json",
    "detail_latency_save_path":"./tmp/detail_latency.json"
}
for "exec", should be orginzed as:
{
    "type":"exec",
    
}


"model_info": a dict, must contatin "type", and other params

"req_type":"normal" or "fixed"