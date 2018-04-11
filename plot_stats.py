import os, json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Collect arguments (if any)
parser = argparse.ArgumentParser()

# Model
parser.add_argument('model', type=int, choices=[1,2,3], help='Which model to choose.')

args = parser.parse_args()

with open(os.path.join('data', 'stats', f'model_stats_0{args.model}.json'), 'r') as f:
    stats = json.load(f)

train = stats['train']
valid = stats['val']
epochs = list(range(1,16))

plt.plot(epochs, train, color='C0', label='Trénovací chyba')
plt.plot(epochs, valid, color='C1', label='Validační chyba')
plt.xticks(epochs)
plt.xlabel("Epochy", fontsize=16)
plt.yticks(range(1100, int(max(train))+40, 25))
plt.ylabel("MAE", fontsize=16)
plt.legend(fontsize=12)
plt.grid(linestyle=':', linewidth=0.5)
plt.savefig(os.path.join('data', 'stats', f'training_0{args.model}.png'))
plt.show()
