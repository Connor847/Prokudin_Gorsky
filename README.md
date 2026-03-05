# Prokudin-Gorsky Image Colorization

Taken in the early 1900s, the Prokudin-Gorsky images capture scenes using three separate RGB exposures on glass plates. This script automatically combines those three black-and-white channel images into a single full-color image using pyramid alignment and border detection.

The pyramid alignment strategy allows large TIFF files to be processed in under a minute while maintaining high alignment quality. Border detection crops each channel prior to alignment, reducing image size and producing a cleaner final result.

| Before | After |
|--------|-------|
| ![Before](assets/emir_before.jpg) | ![After](assets/emir_after.jpg) |



## Usage

### Dependencies
Install the required packages with:
````pip install numpy scikit-image matplotlib```

### Running the Script
Open `colorize_prokudin_gorsky.py` and set the input and output paths at the bottom of the file:
```python
INPUT_PATH = "path/to/your/input.tif"
OUTPUT_PATH = "path/to/your/output.jpg"
```
Then run:
```python colorize_prokudin_gorsky.py```

### Input Format
The script expects a single `.tif` file containing three equal-height grayscale strips stacked vertically in order: Blue, Green, Red. This is the standard format for digitized Prokudin-Gorsky glass plate scans, available from the [Library of Congress](https://www.loc.gov/collections/prokudin-gorsky/).
