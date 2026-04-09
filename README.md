# House Price Prediction

A machine learning project for predicting California house prices using regression models. This project includes exploratory data analysis (EDA), model training, and web applications for predictions.

## Dataset

The project uses the California Housing dataset, which contains information about houses in California districts. Features include:
- Median income
- Housing median age
- Average rooms/bedrooms
- Population
- Latitude/Longitude
- Ocean proximity

## Project Structure

```
house-price-prediction/
├── app.py                    # Flask web application
├── main.py                   # Main training script
├── predict_house_price.py    # Prediction script
├── streamlit_app.py          # Streamlit web application
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── README.md                 # This file
├── house/
│   ├── 1553768847-housing.csv # Dataset
│   └── EDA.ipynb            # Exploratory Data Analysis notebook
├── src/
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   └── components/
│       ├── __init__.py
│       ├── data_ingestion.py
│       ├── data_preprocessing.py
│       ├── model_evaluatiom.py
│       ├── model_trainer.py
│       └── prediction_pipeline.py
├── artifacts/
│   ├── train.csv
│   ├── test.csv
│   ├── training_results.html
│   └── prediction_result.html
└── template/
    └── index.html
```

## Features

- **Exploratory Data Analysis**: Comprehensive analysis with visualizations
- **Model Training**: Multiple regression models with evaluation
- **Web Applications**: Both Flask and Streamlit interfaces
- **Prediction Pipeline**: Modular pipeline for data processing and prediction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tharungandhe/house-price.git
cd house-price-prediction
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main training script:
```bash
python main.py
```

### Running Flask App

```bash
python app.py
```
Access at: http://localhost:5000

### Running Streamlit App

```bash
streamlit run streamlit_app.py
```
Access at: http://localhost:8501

### Making Predictions

Use the prediction script:
```bash
python predict_house_price.py
```

## Exploratory Data Analysis

Open `house/EDA.ipynb` to explore the dataset analysis including:
- Data distribution plots
- Correlation analysis
- Outlier detection
- Feature relationships

## Model Performance

The trained models are evaluated using metrics like:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Technologies Used

- **Python** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization
- **Flask** - Web framework
- **Streamlit** - Web app framework
- **Jinja2** - Template engine

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- California Housing dataset from Kaggle
- Scikit-learn documentation
- Flask and Streamlit communities