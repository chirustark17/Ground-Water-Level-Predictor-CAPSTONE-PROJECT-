# Ground Water Level Predictor

A Capstone Project (PSCS_327) under the Ministry of Jal Shakti, focused on building a software system that predicts and forecasts groundwater levels across regions of India using AI/ML techniques.

## Problem Statement

Groundwater is the backbone of drinking water, agriculture, and industry, but it is rapidly depleting due to over-extraction, urbanization, and climate change. Current monitoring systems (CGWB wells, DWLRs) provide raw data collection but lack forecasting capabilities. Traditional statistical models like ARIMA struggle with the non-linear, multi-variable nature of groundwater systems.

Our solution is a machine learning-based predictive platform that integrates hydrogeological, meteorological, and demographic data to forecast groundwater levels in both time and space, while also filling data gaps.

## Objectives

- Aggregate time-series groundwater level data with relevant factors (rainfall, soil type, temperature, etc.)
- Preprocess, clean, and standardize multi-source datasets
- Implement and train ML/DL models including:
  - Random Forest
  - LSTM
  - XGBoost
  - Hybrid CNN-LSTM (for spatial-temporal learning)
- Evaluate models with metrics like RMSE and MAE
- Develop an interactive dashboard for real-time prediction and visualization
- Ensure the system is scalable, user-friendly, and accessible

## Background & Related Work

Traditional models (ARIMA, regression) show limited predictive accuracy. Recent research highlights the strength of deep learning approaches including LSTM, CNN-LSTM, and ensemble learning methods. This project directly aligns with UN Sustainable Development Goals (SDG 6: Clean Water & Sanitation, SDG 13: Climate Action).

## Datasets

- **Groundwater Levels**: Central Ground Water Board (CGWB) via India-WRIS
- **Meteorological Data**: Indian Meteorological Department (IMD) – rainfall, temperature, humidity
- **Geospatial Attributes** (optional): NBSS & LUP – soil classification, land use
- **Demographic Data**: Census & GIS datasets – population, land utilization

## Innovation & Contribution

- Hybrid CNN-LSTM architecture for spatio-temporal prediction
- Ensemble learning techniques to improve model robustness
- Data imputation algorithms to handle missing values
- Web-based dashboard (Streamlit/Flask/Dash) for visualization & decision support
- Fully open-source workflow on GitHub for reproducibility

## Technology Stack

**Programming Languages**: Python (Pandas, NumPy), R (optional)

**Machine Learning & Deep Learning**: scikit-learn, TensorFlow, Keras

**Database Systems**: PostgreSQL, MongoDB

**Data Visualization**: Matplotlib, Seaborn, Plotly, Power BI

**Web Dashboard**: Streamlit / Flask / Dash

**Development Tools**: VS Code, Git, GitHub

**Hardware Requirements**: 8GB+ RAM, GPU recommended for deep learning models

## Project Timeline

- **Weeks 1–2**: Problem finalization, literature survey
- **Weeks 3–4**: Data collection & preprocessing
- **Weeks 5–6**: Model implementation (baseline ML)
- **Weeks 7–8**: Deep learning models (LSTM, CNN-LSTM)
- **Weeks 9–10**: Dashboard development
- **Weeks 11–12**: Testing, validation, final report & deployment

## Installation & Usage

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/GroundWaterLevelPredictor.git
   cd GroundWaterLevelPredictor
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure database connections and API keys in `config.py`

### Running the Application

1. Train the machine learning models:
   ```bash
   python train_model.py
   ```

2. Launch the web dashboard:
   ```bash
   streamlit run app.py
   ```

3. Access the application at `http://localhost:8501`

## Model Performance

The system implements multiple machine learning algorithms and evaluates their performance using standard metrics:

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)
- Mean Absolute Percentage Error (MAPE)

## Contributing

We welcome contributions to improve the Ground Water Level Predictor. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Kumar, A. "Groundwater level prediction using ML models", IEEE Access, 2021
- Smith, J. "Deep learning for water resource management", IEEE TNNLS, 2022
- Patel, R. "Time-series forecasting of groundwater levels", Springer, 2020

## Team & Acknowledgments

**Team**: Batch PSCS_327

**Project Supervisor**: Dr. Yamanappa (Assistant Professor, CSE, Presidency University)

**Project Coordinators**: Dr. Sampath A K, Dr. Geetha A

**Institution**: Presidency University

**Ministry Partner**: Ministry of Jal Shakti, Government of India

## Contact

For questions or support, please open an issue on GitHub or contact the development team.

---

**Note**: This project is part of an academic capstone program and is intended for research and educational purposes.
