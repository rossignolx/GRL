import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

LOG_DIR = "logs/kgnn"
files = os.listdir(LOG_DIR)

paths = [os.path.join(LOG_DIR, file) for file in files]

no_layers = 3
repeats = 3

model_names = ['ConnectedThreeMalkin', 'TwoMalkin', 'ThreeMalkin', 'ThreeGlobal']
model_names_dict = {name: idx for idx, name in enumerate(model_names)}

accs = [[[[0, 0] for _ in range(repeats)] for _ in range(no_layers)] for _ in range(len(model_names))]
runs = [[[None for _ in range(repeats)] for _ in range(no_layers)] for _ in range(len(model_names))]

for idx, path in enumerate(paths):
    file = os.path.basename(path).rstrip('.txt')
    parts = file.split("-")

    model_name, no_layer = parts[0], parts[-1]
    repeat = int(parts[-3]) - 1

    model_idx = model_names_dict[model_name]

    no_layer = int(no_layer)
    layer_idx = (no_layer - 1) // 2

    val_accs = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines[:-1]:
            parts = line.split(",")
            val_part = parts[-1].split(":")[1]
            val_acc = float(val_part.strip("\n"))
            val_accs.append(val_acc)

    final_line = lines[-1]

    parts = final_line.split(",")
    train_part, test_part = parts[0].strip(" "), parts[2].strip(" ")
    train_acc, test_acc = float(train_part.split(":")[1].strip(" ")), float(test_part.split(":")[1].strip(" "))

    runs[model_idx][layer_idx][repeat] = val_accs

    accs[model_idx][layer_idx][repeat][0] = train_acc
    accs[model_idx][layer_idx][repeat][1] = test_acc

arr = np.array(accs)
mean, std = np.mean(arr, axis=2), np.std(arr, axis=2)

for idx, name in enumerate(model_names):
    for layer_idx in range(no_layers):
        layer = layer_idx * 2 + 1
        train_acc_m, test_acc_m = mean[idx][layer_idx][0], mean[idx][layer_idx][1]
        train_acc_s, test_acc_s = std[idx][layer_idx][0], std[idx][layer_idx][1]
        print("{}, Layer: {}, Train Mean: {}, Train Std: {}".format(name, layer, train_acc_m, train_acc_s))
        print("{}, Layer: {}, Test Mean: {}, Test Std: {}".format(name, layer, test_acc_m, test_acc_s))


rename = {
    'ConnectedThreeMalkin': '3-CM',
    'TwoMalkin': '2-M',
    'ThreeMalkin': '3-M',
    'ThreeGlobal': '3-G'
}

colors = ['b', 'r', 'g']

sns.set_theme()
fig, axes = plt.subplots(2, 2, figsize=(14, 5), sharey=True, sharex=True)

plt.subplots_adjust(wspace=0.055, hspace=0.263)

axes_flat = axes.flat
for idx, name in enumerate(model_names):
    repeat_idx = 1


    axis = axes_flat[idx]
    axis.set_title(rename[name], fontsize=16, fontweight='bold')
    for i in range(no_layers):
        label = 'Layer {}'.format(2 * i + 1)
        s = runs[idx][i][repeat_idx]
        sns.lineplot(s, label=label, ax=axis, legend=False,  palette='deep')
fig.supylabel('Validation Accuracy', fontsize=16, fontweight='bold')
fig.supxlabel('Epochs', fontsize=16, fontweight='bold')

handles, labels = axes_flat[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.65, 0.05), fontsize=12, ncol=3)

plt.tight_layout(pad=1.2)
plt.savefig('model-accuracies.svg', format='svg')




