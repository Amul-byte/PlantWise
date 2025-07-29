# PlantWise

PlantWise is a machine learning project for plant disease detection using deep learning. It
leverages TensorFlow and OpenCV to classify plant diseases from images.

## Project Structure

- `App/` - Application source code
- `model/plantwise_model.h5` - Trained Keras/TensorFlow model
- `Notebooks/` - Jupyter notebooks for experimentation and testing
- `PlantVillage/` - Dataset folders for different plant diseases
- `test_image/` - Test images for inference
- `utils/` - Utility scripts

## Setup

1. Install dependencies:

   ```sh
   pip install -r requirement.txt
   ```

2. Place your `kaggle.json` in the root directory for dataset downloads.

## Usage

- Run the Streamlit app:
  ```sh
  streamlit run Notebooks/app.py
  ```
- Explore and run Jupyter notebooks in the `Notebooks/` directory for model training and testing.

## Requirements

- Python >= 3.12
- See `pyproject.toml` for all Python dependencies.

## Model

The model is stored in `model/plantwise_model.h5` and is used for inference in the app.
