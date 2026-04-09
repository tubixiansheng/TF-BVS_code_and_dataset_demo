# TF-BVS Supplementary Material

## Introduction
This compressed package serves as the supplementary material for the TF-BVS framework. It primarily contains the core code files for the TF-BVS model, the appendix PDF, and instructions for viewing the interactive demo of the breast tumor ultrasound video segmentation dataset.

Specifically, upon extracting this package, you will find the following files:
- **`README.md`**: The current instruction document, containing an overview of the attachments and a tutorial on running the dataset visualization page.
- **`TF-BVS_Code/`** directory: Contains the code implementation of the core modules in the TF-BVS model.
- **`Supplementary_Material.pdf`**: The appendix document, containing detailed descriptions of the dataset, experimental setups, additional ablation studies, and hyperparameter sensitivity analysis.

**🌐 Anonymous Repository Access:**
To facilitate interactive review for the reviewers, we have provided a link to an anonymous repository within `Supplementary_Material.pdf`. This repository not only contains all the aforementioned code and appendix contents but also hosts the **online dataset visualization page** and the **dataset sample source files**.
* **Anonymous Repository Link**: [https://anonymous.4open.science/r/TF-BVS_code_and_dataset_demo-70D7](https://anonymous.4open.science/r/TF-BVS_code_and_dataset_demo-70D7)

---

## File Structure Overview

To clearly illustrate the contents in different environments, the file structures for the "Local Extracted Package" and the "Online Anonymous Repository" are listed separately below:

### 1. File Structure of the Local Extracted Package
```plaintext
Supplementary_Material
│
├── README.md                    # Supplementary material instructions (this file)
├── TF-BVS_Code/                 # Core code folder for the TF-BVS model
│   ├── Backbone/                # Cross-frame Temporal Backbone module code
│   ├── Branch/                  # Frame-wise Temporal Branch module code
│   └── LV2TOT/                  # LV2TOT module code
│
└── Supplementary_Material.pdf   # Appendix file containing supplementary details
```


### 2. File Structure of the Anonymous Repository
```
TF-BVS_code_and_dataset_demo-70D7
│
├── assets/                      # Static resources and source files for the dataset demo
├── css/                         # Static resources for the dataset demo
├── js/                          # Static resources for the dataset demo
├── index.html                   # Main page for interactive dataset visualization
│
├── TF-BVS_Code/                 # Core code folder for the TF-BVS model
│   ├── Backbone/                # Cross-frame Temporal Backbone module code
│   ├── Branch/                  # Frame-wise Temporal Branch module code
│   └── LV2TOT/                  # LV2TOT module code
│
└── Supplementary_Material.pdf   # Appendix file containing supplementary details
```

---

## How to Open the Dataset Demo
To access the dataset visualization demo, please visit the anonymous repository link provided above. You can choose to run it directly online or download it for local execution:
1. Online Preview (Recommended): In the anonymous repository file list, click on the index.html file, and then click the View raw button in the top right corner to run the interactive demo directly online. (Note: If the page data does not load completely or displays incorrectly, please refresh the page and try again.)
2. Local Execution: Alternatively, you can download the complete repository folder to your local computer and open the index.html file using any web browser that supports HTML. This will similarly load and display the interactive dataset demo.
💡 Interaction Guide:
   - On the demo page, you can drag the index slider at the bottom to browse the breast ultrasound videos frame by frame while intuitively viewing their corresponding tumor mask annotations.
   - **Recommended Browsers**:
      - Chrome
      - Edge
      - Firefox
   - **Offline Support**: If you choose the "Local Execution" method, the demo fully supports offline browsing and does not require an internet connection.

---

## Appendix Details (Supplementary_Material.pdf)
- **File Path**: The appendix file is located in the project root directory with the filename `Supplementary_Material.pdf`.
- **Main Contents**:
  1. Detailed description of the multi-modal ultrasound dataset.
  2. Experimental setups for adapting natural-scene baseline models, along with additional ablation study results.
  3. Sensitivity analysis of core hyperparameters.

---

## Notes
- Network Connection: Running the dataset demo locally does not require an internet connection; however, previewing the demo directly online via the anonymous repository requires a stable network connection.
- Future Work: We will continue to expand and refine this multi-modal dataset, with a particular focus on supplementing the clinically scarce high-risk grade (High BI-RADS grade) cases to further enhance the model's generalization capability and clinical value.