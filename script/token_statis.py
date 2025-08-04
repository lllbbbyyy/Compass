import json
import matplotlib.pyplot as plt
import numpy as np

input_file = '../config/govreport_input_token_lens.json'
output_file = '../config/govreport_output_token_lens.json'

with open(input_file, 'r') as f:
    input_token_list = json.load(f)
with open(output_file, 'r') as f:
    output_token_list = json.load(f)

print("Number of samples:", len(input_token_list))
print("Max input sequence length:", max(input_token_list))
print("Max output sequence length:", max(output_token_list))
print("Min input sequence length:", min(input_token_list))
print("Min output sequence length:", min(output_token_list))

print("Average input sequence length:", sum(input_token_list) / len(input_token_list))
print("Average output sequence length:", sum(output_token_list) / len(output_token_list))

# Create figure with two subplots (no longer sharing Y-axis)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# Set main figure title
fig.suptitle('Token Length Distribution Analysis', fontsize=16)

# Define box properties
boxprops = dict(linestyle='-', linewidth=1.5, color='black')
medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
whiskerprops = dict(linestyle='-', linewidth=1.5, color='black')
capprops = dict(linestyle='-', linewidth=1.5, color='black')
meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='gold', markersize=8)

# Input token boxplot (left plot)
input_plot = axes[0].boxplot(input_token_list, 
               boxprops=boxprops,
               medianprops=medianprops,
               whiskerprops=whiskerprops,
               capprops=capprops,
               showmeans=True, 
               meanprops=meanprops,
               patch_artist=True)

# Calculate stats for input
input_median = np.median(input_token_list)
input_mean = np.mean(input_token_list)
input_max = np.max(input_token_list)
input_min = np.min(input_token_list)
input_q1 = np.percentile(input_token_list, 25)
input_q3 = np.percentile(input_token_list, 75)

# Add text stats for input
stats_text = f'Dataset: {input_file.split("/")[-1].split("_")[0]}\n\n' \
             f'Min: {input_min}\n' \
             f'1st Quartile: {input_q1:.1f}\n' \
             f'Median: {input_median:.1f}\n' \
             f'3rd Quartile: {input_q3:.1f}\n' \
             f'Max: {input_max}\n' \
             f'Mean: {input_mean:.1f}\n' \
             f'Sample Count: {len(input_token_list)}'

axes[0].text(0.95, 0.95, stats_text,
             transform=axes[0].transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)

# Set input plot title and labels
axes[0].set_title('Input Sequence Length Distribution', fontsize=14)
axes[0].set_ylabel('Token Count', fontsize=12)
axes[0].set_xlabel('Input Sequences', fontsize=12)
axes[0].set_xticks([])
axes[0].grid(True, linestyle='--', alpha=0.7)

# Output token boxplot (right plot)
output_plot = axes[1].boxplot(output_token_list, 
               boxprops=boxprops,
               medianprops=medianprops,
               whiskerprops=whiskerprops,
               capprops=capprops,
               showmeans=True, 
               meanprops=meanprops,
               patch_artist=True)

# Calculate stats for output
output_median = np.median(output_token_list)
output_mean = np.mean(output_token_list)
output_max = np.max(output_token_list)
output_min = np.min(output_token_list)
output_q1 = np.percentile(output_token_list, 25)
output_q3 = np.percentile(output_token_list, 75)

# Add text stats for output
stats_text = f'Dataset: {output_file.split("/")[-1].split("_")[0]}\n\n' \
             f'Min: {output_min}\n' \
             f'1st Quartile: {output_q1:.1f}\n' \
             f'Median: {output_median:.1f}\n' \
             f'3rd Quartile: {output_q3:.1f}\n' \
             f'Max: {output_max}\n' \
             f'Mean: {output_mean:.1f}\n' \
             f'Sample Count: {len(output_token_list)}'

axes[1].text(0.95, 0.95, stats_text,
             transform=axes[1].transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
             fontsize=10)

# Set output plot title and labels
axes[1].set_title('Output Sequence Length Distribution', fontsize=14)
axes[1].set_ylabel('Token Count', fontsize=12)  # Now each has its own Y-axis
axes[1].set_xlabel('Output Sequences', fontsize=12)
axes[1].set_xticks([])
axes[1].grid(True, linestyle='--', alpha=0.7)

# Set different Y-axis scales for input and output (adapt to each dataset)
# Get current Y-limits for input plot
input_y_min, input_y_max = axes[0].get_ylim()
# Use a 10% buffer above the max value and 10% below min value (at least 0)
input_new_min = max(0, input_y_min - 0.1 * (input_y_max - input_y_min))
input_new_max = input_y_max + 0.1 * (input_y_max - input_y_min)
axes[0].set_ylim([input_new_min, input_new_max])

# Same for output plot
output_y_min, output_y_max = axes[1].get_ylim()
output_new_min = max(0, output_y_min - 0.1 * (output_y_max - output_y_min))
output_new_max = output_y_max + 0.1 * (output_y_max - output_y_min)
axes[1].set_ylim([output_new_min, output_new_max])

# Customize box colors to match plot meaning
for box in input_plot['boxes']:
    box.set(facecolor='#FFD700')  # Gold for input
for box in output_plot['boxes']:
    box.set(facecolor='#1E90FF')  # DodgerBlue for output

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
plt.savefig(f'{input_file.split("/")[-1].split("_")[0]}_token_lengths.png', dpi=150, bbox_inches='tight')
plt.show()

# Additional printed statistics
print("\nDetailed Statistics:")
print("Input: Min={}, Max={}, Mean={:.1f}, Median={:.1f}, Q1={:.1f}, Q3={:.1f}".format(
    input_min, input_max, input_mean, input_median, input_q1, input_q3))
print("Output: Min={}, Max={}, Mean={:.1f}, Median={:.1f}, Q1={:.1f}, Q3={:.1f}".format(
    output_min, output_max, output_mean, output_median, output_q1, output_q3))