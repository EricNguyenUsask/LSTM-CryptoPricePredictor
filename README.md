# LSTM-CryptoPricePredictor

## Overview

This project aims to predict cryptocurrency prices using a Long Short-Term Memory (LSTM) model. The focus is on understanding how machine learning can be applied to cryptocurrency price prediction while highlighting the limitations of AI in capturing complex human-driven market fluctuations.

---

## Dataset

The data used for this project is sourced from Kaggle: [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data). This dataset contains historical price data for Bitcoin, including close prices used to train the model. Data is resampled to daily averages for better noise reduction.

---

## Methodology

1. **Preprocessing**: Resampled the data to daily intervals and normalized it using MinMaxScaler.
2. **Time-Series Data**: Created sequences of 60 days as input and the subsequent day as the target for training.
3. **Model Architecture**: Utilized an LSTM model with 50 hidden units, followed by a dense layer for predictions.
4. **Training**: Trained the model using Mean Squared Error (MSE) loss and the Adam optimizer over 20 epochs with a batch size of 64.
5. **Evaluation**: Predicted prices on the test set and compared them to actual prices using the Mean Absolute Error (MAE).

---

## Results

- **MAE**: The model achieved an error of **$1287** for every **$10,000** of Bitcoin.
- The AI struggled to accurately predict Bitcoin price fluctuations due to external factors such as:
  - Social media influence (e.g., tweets from public figures like Donald Trump).
  - Major global events (e.g., wars, economic crises).
  - Market sentiment and investor behavior.

These factors cannot be captured effectively by historical price data alone, making cryptocurrency prediction inherently challenging for AI.

---

## Limitations

- **External Factors**: Machine learning models cannot account for non-quantifiable events, such as geopolitical developments, regulatory announcements, or influential social media posts.
- **Model Complexity**: Increasing or decreasing the complexity of the model (e.g., switching from LSTM to simpler or more advanced architectures) did not significantly improve the results. This limitation emphasizes that the data itself lacks critical context for accurate predictions.

---

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/crypto-price-prediction.git
   cd crypto-price-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) and place it in the `data/` directory.
4. Run the project:
   ```bash
   python crypto_price_prediction_lstm.py
   ```

---

## Conclusion

This project demonstrates the potential and limitations of AI in cryptocurrency price prediction. While the model can learn from historical data patterns, the inherent unpredictability of the market, driven by human behavior and external events, imposes significant constraints on prediction accuracy.

---

## Future Work

- Explore integrating sentiment analysis from news or social media to improve predictions.
- Investigate ensemble models combining multiple data sources.
- Apply transfer learning with more diverse financial datasets.
