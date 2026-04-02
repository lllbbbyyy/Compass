import json

file_name='fixed_avg_govreport_input_token_lens.json'
smaple_num=17517
seq_len=9652

with open(file_name, 'w') as f:
    json.dump([seq_len]*smaple_num, f)