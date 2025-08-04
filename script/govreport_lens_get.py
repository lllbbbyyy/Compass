import json
from datasets import load_dataset
from transformers import AutoTokenizer
import tiktoken
import matplotlib.pyplot as plt
from scipy.stats import gamma
import pickle
from tqdm import tqdm

# load GovReport dataset
dataset = load_dataset("ccdv/govreport-summarization", split="train")

encoding = tiktoken.encoding_for_model('gpt-4')
input_token_lengths = []
output_token_lengths = []

for i, sample in tqdm(enumerate(dataset)):
    input_text = sample['report']
    output_text = sample['summary']

    input_tokens = encoding.encode(input_text)
    output_tokens = encoding.encode(output_text)

    input_token_lengths.append(len(input_tokens))
    output_token_lengths.append(len(output_tokens))

print("min input token length:", min(input_token_lengths))
print("max input token length:", max(input_token_lengths))
print("min output token length:", min(output_token_lengths))
print("max output token length:", max(output_token_lengths))
print("avg input token length:", sum(input_token_lengths) / len(input_token_lengths))
print("avg output token length:", sum(output_token_lengths) / len(output_token_lengths))

# save to json file
with open('govreport_input_token_lens.json', 'w') as f:
    json.dump(input_token_lengths, f)
with open('govreport_output_token_lens.json', 'w') as f:
    json.dump(output_token_lengths, f)

# draw histogram of token lengths
plt.hist(input_token_lengths, bins=1000, alpha=0.5, label='input Token lens')
plt.hist(output_token_lengths, bins=1000, alpha=0.5, label='output Token lens')
plt.legend()
plt.xlabel('Token lens')
plt.ylabel('freq')
plt.title('input and output Token lens distri')
plt.savefig('govreport_token_length_distribution.png')
plt.close()

# fit gamma distribution to the token lengths
input_params = gamma.fit(input_token_lengths, floc=0)
output_params = gamma.fit(output_token_lengths, floc=0)

# save the distribution parameters to a pickle file
with open('../config/govreport_distribution_params.pkl', 'wb') as f:
    pickle.dump({'input_params': input_params, 'output_params': output_params}, f)

print("distribution params saved to 'govreport_distribution_params.pkl'")
print("histogram saved to 'govreport_token_length_distribution.png'")
