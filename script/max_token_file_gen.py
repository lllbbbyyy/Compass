import json

file_name='fixed_sharegpt_input_token_lens.json'
smaple_num=20000
seq_len=1257

with open(file_name, 'w') as f:
    json.dump([seq_len]*smaple_num, f)