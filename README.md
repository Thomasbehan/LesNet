<img src="/skinvestigatorai/static/logo.png" align="right" width="100" height="100" />

# LesNet  ![View SkinVestigatorAI on GitHub](https://img.shields.io/github/stars/Thomasbehan/LesNet?color=232323&label=LesNet&logo=github&labelColor=232323)
![Precision Score](https://img.shields.io/badge/Precision-94.22%25-darkgreen)
![Recall Score](https://img.shields.io/badge/Recall-80.43%25-darkgreen)
![Accuracy Score](https://img.shields.io/badge/Accuracy-86.17%25-darkgreen)
![Loss Score](https://img.shields.io/badge/Loss-0.4621-blue)
![GitHub license](https://img.shields.io/github/license/Thomasbehan/LesNet) [![Actions Status](https://github.com/Thomasbehan/LesNet/workflows/Automated%20Testing/badge.svg)](https://github.com/Thomasbehan/LesNet/actions)
[![Actions Status](https://github.com/Thomasbehan/LesNet/workflows/CodeQL/badge.svg)](https://github.com/Thomasbehan/LesNet/actions)

> LesNet is an open-source project for deep learning-based skin cancer detection. It aims to create a reliable tool and foster community involvement in critical AI problems. The repository includes code for data preprocessing, model building, and performance evaluation. Contribute and shape the future of skin cancer detection.

[![Demo](https://img.shields.io/badge/-Live_Demo-black?style=for-the-badge&logo=render)](https://lesnet.onrender.com/) 

<sub><i>Please note that the application enters a dormant state when not in use to conserve resources. This means it might take a moment to warm up when you first access the site. Any initial slow down will ease after a moment. Thank you for your patience.
</i></sub>

## Table of Contents
- [Getting Started](#getting-started)
- [Data](#data)
- [Model](#model)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Citation](#citation)
- [Disclaimer](#disclaimer)
 
# Getting Started
<img src="screenshot.png" align="center" />

These instructions will help you set up the project on your local machine for development and testing purposes. See [deployment](#deployment) for notes on deploying the project on a live system.

## Quick Setup for Development

To quickly set up LesNet for development, follow these steps
(Requires Python >=3.9<=3.11):

1. **Upgrade Your Packaging Tools:**
   Ensure your `pip` and `setuptools` are up-to-date by running:
   ```bash
   python -m pip install --upgrade pip setuptools
   ```

2. **Install LesNet:**
   In the project directory, install the project in editable mode with:
   ```bash
   python -m pip install -e .[testing]
   ```

3. **Run the Application:**
   Start the application with auto-reloading using:
   ```bash
   pserve development.ini --reload
   ```

## Running the Tests and Linting
[![Actions Status](https://github.com/Thomasbehan/LesNet/workflows/Automated%20Testing/badge.svg)](https://github.com/Thomasbehan/LesNet/actions)
[![Actions Status](https://github.com/Thomasbehan/LesNet/workflows/CodeQL/badge.svg)](https://github.com/Thomasbehan/LesNet/actions)

### Running the Tests
To run the tests, run the following command:
```bash
python -m pytest
```

### Running the Linter
To run the linter, run the following command:
```bash
python -m ruff check
```

## Model Downloader

To download and prepare a specific model for use, you can use the `download_model.py` script located in the `commands` directory. This script accepts the model identifier as an argument.

### Usage

Run the following command from the root of the project directory:

```bash
python .\commands\download_model.py -m <model_id>
```
### Available Models
Here is a list of all the available models you can download using the script:

* M-0003: Simple Testing (Legacy).
* M-0015: Best Model (Legacy).
* M-0015s: Fastest Model (Legacy)
* M-0310: Best Model.
* M-0310s: Efficient Model.

* Example: 
```bash
python .\commands\download_model.py -m M-0310s
```

## Data
The DataScraper tool within this application is designed to download and preprocess skin lesion images. The M-3.1 dataset is 837,628 images.

### Data Source
The dataset used for training the model is sourced from the International Skin Imaging Collaboration (ISIC) Archive. The ISIC Archive is a large-scale resource for skin image analysis, providing open access to a wide variety of images for the development and evaluation of automated diagnostic systems.

For more information about the ISIC Archive and to access the data, visit [ISIC Archive](https://www.isic-archive.com).

### Data Organization
The images are organized into three folders:

1. `data/train`: Contains all images, which are used for training the model.
2. Images are placed in folders with their label as its name, for example `data/train/melanoma`

## Model
The `SVModel` model employs a sophisticated deep learning architecture based on InvceptionV3 but tailored for skin lesion classification. 
To learn more, Visit [the model section of the wiki](https://github.com/Thomasbehan/LesNet/wiki#model)

## Performance
The updated model demonstrates significant improvements in its ability to classify skin lesions accurately, achieving an accuracy of 84% and a loss of 0.23 on the testing dataset. The model's sensitivity, specificity, precision, and F1 score have also seen considerable enhancements, with the following scores reported on the testing dataset:

- Recall: 80.43%
- Precision: 94.22%
- Accuracy: 86.17%
- Loss: 0.4621


### Targets

| Metric            | Target Range  | Progress                                                                      |
|-------------------|---------------|-------------------------------------------------------------------------------|
| **Loss**          | Close to 0    | ![Progress](https://progress-bar.dev/94/?scale=0..100&title=progress&suffix=) |
| **Accuracy**      | 85% - 95%     | ![Progress](https://progress-bar.dev/86/?scale=85..95&title=progress&suffix=) |
| **Precision**     | 80% - 90%     | ![Progress](https://progress-bar.dev/94/?scale=80..90&title=progress&suffix=) |
| **Recall**        | 85% - 95%     | ![Progress](https://progress-bar.dev/80/?scale=85..95&title=progress&suffix=) |

## Contributing
contributions to LesNet are welcome! For guidelines on contributing, please read [CONTRIBUTING.md](CONTRIBUTING.md). By participating in this project, you agree to abide by its terms.

## License
LesNet is released under the Mozilla Public License 2.0 (MPL 2.0). For more details, see the [LICENSE](License) file.

## Acknowledgments
Gratitude to the International Skin Imaging Collaboration (ISIC) for providing access to their extensive archive of skin lesion images, which has been instrumental in the development and refinement of this model.

## References
- International Skin Imaging Collaboration (ISIC). The ISIC Archive. https://www.isic-archive.com

## Citation
For academic and research use, please cite this work as follows:

"LesNet: A Deep Learning-Based Skin Cancer Detection Tool, available at: https://github.com/Thomasbehan/LesNet", 2024.

## Disclaimer
LesNet is not intended for clinical diagnosis or medical use. It is a research tool aimed at fostering developments in the field of automated skin lesion analysis. Always consult a qualified healthcare provider for medical advice and diagnosis.

