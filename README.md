# IPL Score Prediction Web-App 
[Live Link] (https://knowiplscore.streamlit.app/)   
[Colab Notebook] (https://colab.research.google.com/drive/1Lctq2eHBAh1qXoqMLeT98TfhoNV322Ja#scrollTo=SfDDg1txjRv2)

This repository contains an IPL Score Prediction Web-App built with Python. It allows users to predict the final score of an Indian Premier League (IPL) cricket match based on match and team statistics. The project leverages machine learning techniques to provide real-time score predictions with an easy-to-use web interface.

## Features

- **Predicts IPL match scores** using machine learning models trained on historical data.
- **User-friendly web interface** for interactive predictions.
- **Custom input** for teams, match situation, overs, wickets, and other game stats.
- Built with **Python**, leveraging libraries like scikit-learn, pandas, and Flask or Streamlit.

## Directory Structure

```
MLG_AAT/
├── .idea/
├── .gitattributes
├── app.py            # Main web application file
├── ipl_data.csv      # Dataset used for model training
├── ipl_model.py      # Machine learning model training and prediction logic
├── ipl_models.zip    # Pretrained model(s)
├── requirements.txt  # Python package dependencies
```

## Getting Started

### Prerequisites

- Python 3.7 or above
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aditya-is22/MLG_AAT.git
   cd MLG_AAT
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Unzip pre-trained models:**
   If required, extract `ipl_models.zip` to access trained model files.

## Usage

1. **Run the web app:**
   ```bash
   python app.py
   ```
   (Ensure you have Flask/Streamlit set up as per `app.py`.)

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```
   or as specified by the Flask/Streamlit output.

3. **Input match details** using the web form and get instant score predictions.

## Files Explained

- **app.py:** Contains the web application code (Flask or Streamlit) for user interaction.
- **ipl_model.py:** Houses model training, loading, and prediction functions.
- **ipl_data.csv:** Historical IPL match data used for training the models.
- **ipl_models.zip:** Compressed file with machine learning model(s) ready for deployment.
- **requirements.txt:** All necessary dependencies for running the app.

## Model Training

To retrain the machine learning model(s):

```bash
python ipl_model.py
```

This will train the model on `ipl_data.csv`. The output model will be saved and can be used by `app.py` for predictions.

## Contributing

Feel free to open issues or make pull requests. For significant changes, please open an issue first to discuss what you would like to change.

## License

This project is open-source and available under the MIT License.

## Disclaimer

This app is for educational and demonstration purposes. The prediction is based on historical data and model assumptions; actual match results may vary.

Enjoy predicting IPL scores!

https://github.com/aditya-is22/MLG_AAT
