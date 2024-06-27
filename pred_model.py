import numpy as np
import pandas as pd
from keras.src.legacy.saving import legacy_h5_format
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("cleaned_dataset.csv")

# Tokenizer and sequence preparation
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Description'].values)

# Encode the "Stock Code" and "Country"
stock_code_encoder = LabelEncoder()
data['StockCodeEncoded'] = stock_code_encoder.fit_transform(data['StockCode'])

country_encoder = LabelEncoder()
data['CountryEncoded'] = country_encoder.fit_transform(data['Country'])

# Load the model
model = legacy_h5_format.load_model_from_hdf5("text_model.h5", custom_objects={'mse': 'mse'})

# Sample new data for prediction
new_data = pd.DataFrame({
    'Description': ['WHITE HANGING HEART T-LIGHT HOLDER', 'white metal lantern']
})

# Preprocess the new data
new_X = tokenizer.texts_to_sequences(new_data['Description'].values)
new_X = pad_sequences(new_X, maxlen=100)

# Make predictions
predictions = model.predict(new_X)

# Debugging: Print the content and shape of the predictions
print(f"Predictions content: {predictions}")
print(f"Predictions shape: {predictions.shape}")

# Reshape predictions if necessary (assuming (2, 1) shape)
if predictions.shape == (2, 1):
    predictions = np.squeeze(predictions, axis=-1)  # Remove the last dimension
elif predictions.ndim == 2:
    pass  # No reshape needed if already in the expected shape
else:
    raise ValueError("Unexpected prediction shape: " f"{predictions.shape}")

# Assuming the model outputs are as expected, proceed with decoding
predicted_stock_code = np.argmax(predictions[:, 0])  # Assuming the first column is for stock code
predicted_country = np.argmax(predictions[:, 1])  # Assuming the second column is for country

# Decode predictions
predicted_stock_code = stock_code_encoder.inverse_transform([predicted_stock_code])[0]
predicted_country = country_encoder.inverse_transform([predicted_country])[0]

# Update the new_data DataFrame with predictions
new_data['PredictedStockCode'] = predicted_stock_code
new_data['PredictedCountry'] = predicted_country

# Display predictions
print(new_data[['Description', 'PredictedStockCode', 'PredictedCountry']])
