# TF-BVS Supplementary Material

## Introduction
This project contains supplementary materials for the TF-BVS framework, including an interactive demo for the breast tumor ultrasound video segmentation dataset, core code files, and an appendices PDF. Specifically, it includes:
- **Dataset Demo**: A static file for interactively viewing examples of ultrasound videos and their mask annotations from the dataset.
- **TF-BVS Core Code**: Includes the implementation of important modules in the TF-BVS model.
- **Appendix PDF**: Contains detailed descriptions of the dataset, experimental settings, additional ablation studies, and hyperparameter sensitivity analysis.

---

## File Structure
```plaintext
TF-BVS_code_and_dataset_demo-70D7
│
├── assets/                      # Static folder for dataset demo
├── index.html                   # Main page for viewing the dataset
│
├── TF-BVS_Code/                 # Core code folder for the TF-BVS model
│   ├── Backbone/                # Cross-frame Temporal Backbone module
│   ├── Branch/                  # Frame-wise Temporal Branch module
│   └── LV2TOT/                  # LV2TOT module
│
└── Supplementary_Material.pdf  # Appendix file with supplementary materials
```

---

## How to Open the Dataset Demo
1. Online Preview (Recommended): In the anonymous repository, click on the `index.html` file, then click **View raw** in the top right corner of the page to run the interactive demo online. *(Note: If the page data does not load completely, please refresh the page and try again.)
2. Local Execution: Alternatively, download the complete repository folder to your local machine, and open the `index.html` file using any web browser that supports HTML. This will load and display the interactive dataset demo.
   - On the demo page, you can drag the index slider to browse the ultrasound video frames one by one while viewing their corresponding mask annotations.
   - **Recommended Browsers**:
      - Chrome
      - Edge
      - Firefox
   - **Offline Support**: The demo works offline and does not require an internet connection.

---

## Appendix Details
- **File Path**: The appendix file is located in the project root directory with the filename `Supplementary_Material.pdf`.
- **Main Contents**:
  1. Detailed description of the dataset.
  2. Experimental setups and additional ablation studies.
  3. Hyperparameter sensitivity analysis.

---

## Notes
- The dataset demo does not require an internet connection.
- In the future, we will continue to expand and refine the dataset, with a particular focus on supplementing the scarce high-risk grade cases.