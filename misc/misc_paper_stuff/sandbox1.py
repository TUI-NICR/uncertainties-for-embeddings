
from matplotlib import pyplot as plt
import numpy as np


results = []
mAP_list = []
eps_list = []
with open("all_hyper.log", "r") as f:
    for line in f:
        if line[:14] == "best epsilon =":
            mAP = float(line.split("mAP = ")[1].strip())
            eps = float(line.split(" ;")[0].split("= ")[1])
            results.append((mAP, eps))
            mAP_list.append(mAP)
            eps_list.append(eps)

print(results)
print(f"{mAP_list =}")
print(f"{eps_list =}")

plt.plot(np.array(mAP_list), np.array(eps_list), "x")
plt.xlabel("mAP")
plt.ylabel("epsilon")
plt.title("Hyperparameter search results for different UAL runs")
#plt.plot([x_mean, x_mean], [0, np.max(y) * 0.05], 'k-')
plt.savefig("mAP_vs_eps.png")

"""
- hyper.log zeilenweise einlesen
- die mAP Werte für die jeweiligen epsilons ablegen, jeweils pro Run
- Runs rausfiltern, die <85 als beste mAP haben (damit die es nicht verzerren)
- for jeden epsilon wert berechnen, wie die durchschnittliche mAP über alle Runs ist 
- anhand dieser Durchschnittswerte das beste Epsilon bestimmen
- vergleichen mit der durchschnittlichen mAP -> im Durschnitt eine Verbesserung?
"""