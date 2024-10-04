## Environment Setup

To set up the environment and install the necessary dependencies, please follow these steps.

### Prerequisites

Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### Steps

1. **Create a new environment** using the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
2. **Activate the environment**:
    ```bash
    conda activate openml-tags
    ```

3. **Verify the environment** is working as expected by running:
    ```bash
    conda list
    ```

At this point, all required packages should be installed, and you can start using the repository and running the notebooks in the `notebooks` directory.