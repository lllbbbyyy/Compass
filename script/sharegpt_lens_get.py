import json
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import pickle
from tqdm import tqdm

# load tiktoken encoding for GPT-4
encoding = tiktoken.encoding_for_model('gpt-4')

with open('../ShareGPT52K/old/sg_52k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# used to store input and output token lengths
input_token_lengths = []
output_token_lengths = []

# track the role of the last message to differentiate between input and output
flag=0
cnt=0
pre_content=''
pre_role=''
for conversation in tqdm(data):
    messages = conversation['conversations']
    for i in range(len(messages)):
        msg = messages[i]
        role = msg['from']
        content = msg['value']

        # calculate token length
        tokens = encoding.encode(content,disallowed_special=())
        token_length = len(tokens)+1

        if role == 'human':
            if flag==1:
                input_token_lengths.pop()
            input_token_lengths.append(token_length)
            flag=1
        elif role == 'gpt':
            if flag==0:
                pass
            else:
                output_token_lengths.append(token_length)
            flag=0


print('input token number: ',len(input_token_lengths),'min: ',min(input_token_lengths),'max: ',max(input_token_lengths))
print('output token number',len(output_token_lengths),'min: ',min(output_token_lengths),'max: ',max(output_token_lengths))

with open('input_token_lens.json','w') as f:
    json.dump(input_token_lengths,f)
with open('output_token_lens.json','w') as f:
    json.dump(output_token_lengths,f)

# draw histogram of token lengths
plt.hist(input_token_lengths, bins=1000, alpha=0.5, label='input Token lens')
plt.hist(output_token_lengths, bins=1000, alpha=0.5, label='output Token lens')
plt.legend()
plt.xlabel('Token lens')
plt.ylabel('freq')
plt.title('input and output Token lens distri')
plt.savefig('sharegpt_token_length_distribution.png')
plt.close()

# use gamma distribution to fit the token lengths
input_params = gamma.fit(input_token_lengths, floc=0)
output_params = gamma.fit(output_token_lengths, floc=0)

# save the distribution parameters to a pickle file
with open('../config/sharegpt_distribution_params.pkl', 'wb') as f:
    pickle.dump({'input_params': input_params, 'output_params': output_params}, f)

print("distribution params saved to 'sharegpt_distribution_params.pkl'")
print("histogram saved to 'sharegpt_token_length_distribution.png'")
