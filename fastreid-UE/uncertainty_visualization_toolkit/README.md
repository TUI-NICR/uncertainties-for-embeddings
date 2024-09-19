# Uncertainty Visualization Toolkit

This toolkit contains the Uncertainty Ranking visualizer and the Uncertainty Score Histogram Visualizer, as well as a utility to sort distractors into sub-categories.

## General Usage
- Run the server using `python server.py`
- navigate to [localhost:8000](localhost:8000) in your browser

## Uncertainty Ranking Visualizer

### Usage
- Make sure the **Market Path** is set correctly.
- Move `uncertain_images.json` files to `data/` or point **the data path** to your results folder.
  - All files in the indicated folder will be read as well as all files named `uncertain_images.json` in any subfolder of it.
- Open [localhost:8000/ranking_visualizer.html](localhost:8000/ranking_visualizer.html).
- Select which runs to display and press `Render`.
  - NOTE: you can toggle groups as one or expand them and select individual runs.
- Select number of least/medium/most uncertain images and which sets and functions to disable.
  - Render is bound to onchange but there is also the button.


## Uncertainty Score Histogram Visualizer
### Usage
- Make sure the raw model outputs are in this folder in a file called `raw_model_outputs.json`
- Open [localhost:8000/histogram_visualizer.html](localhost:8000/histogram_visualizer.html) in your browser
- Select what Score function to compute for which uncertainty type over which subset (or subsets) and with how many bins in the histogram
- Optionally select a window of relevant score values
- Press `Render`

NOTE: on first render the json file is being loaded which may take several minutes. Refer to server console output for more information.

NOTE: in addition to the histograms, the mean and standard deviation is plotted above them.

### Options
- Set: Q, G, D1, D2, D3, D4 (multiple as comma-separated list possbile, additionals are drawn transparently)
- Unc.-type: model, data, dist, DD (= data / dist)
- Window: Any floats (using dot as decimal separator). If left empty, the min/max of scores is used. Window can be disabled without deleting the numbers using the checkbox
- Score Function: L1, L2, avg, min, max, entropy
- Density: Passed to `matplotlib.pyplot.hist`. From their documentation: If True, draw and return a probability density: each bin will display the bin's raw count divided by the total number of counts and the bin width `(density = counts / (sum(counts) * np.diff(bins)))`, so that the area under the histogram integrates to 1 (`np.sum(density * np.diff(bins)) == 1`).

### Raw model outputs format
```python
raw_data = {
  "data": {
    "0000_c1s1_013251_01.jpg": {
      "mean_vector": [0.1, 0.2, ...],                 # Embedding Vector
      "variance_vector": [0.1, 0.2, ...],             # Data Uncertainty
      "variance_of_mean_vector": [0.1, 0.2, ...],     # Model Uncertainty
      "variance_of_variance_vector": [0.1, 0.2, ...]  # Distributional Uncertainty
    },
    ...
  },
  "sets": {
    "Q": ["0000_c1s1_013251_01.jpg", ...],
    "G": ...,
    "D1": ...,
    "D2": ...,
    "D3": ...,
    "D4": ...
  },
  "glossary": {...} # optional explanation of keys used
}
```


## Distractor Sub-categorization Utility

### Usage
- Make sure Market dataset is in this folder
- Open [localhost:8000/distractor_sorting.html](localhost:8000/distractor_sorting.html) in your browser
- For a given image, press Numpad 1-5 to categorize. Adjust as needed. E.g.:
  1. \>50% of a human is visible
  2. only fragments of a human are visible
  3. only non-human objects are visible
  4. only background / unrecognizable
  5. not categorized (for now)

The inputs are saved into `distraction_levels.txt`. Each line contains the path of the image followed by a comma followed by the number.
This file can be placed in `fastreid/evaluation` to use it in fastreid evaluation.