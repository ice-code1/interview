import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, SpatialDropout1D, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("cleaned_dataset.csv")

# Tokenizer and sequence preparation
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Description'].values)
X = tokenizer.texts_to_sequences(data['Description'].values)
X = pad_sequences(X, maxlen=100)

# Encode the "Stock Code" and "Country"
stock_code_encoder = LabelEncoder()
data['StockCodeEncoded'] = stock_code_encoder.fit_transform(data['StockCode'])

country_encoder = LabelEncoder()
data['CountryEncoded'] = country_encoder.fit_transform(data['Country'])

# Prepare the target variables
y_stock_code = data['StockCodeEncoded'].values
y_unit_price = data['UnitPrice'].values
y_country = data['CountryEncoded'].values

# Define the model
input_layer = Input(shape=(100,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=100)(input_layer)
dropout_layer = SpatialDropout1D(0.2)(embedding_layer)
lstm_layer = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(dropout_layer)

# Output layers
stock_code_output = Dense(len(stock_code_encoder.classes_), activation='softmax', name='stock_code_output')(lstm_layer)
unit_price_output = Dense(1, activation='linear', name='unit_price_output')(lstm_layer)
country_output = Dense(len(country_encoder.classes_), activation='softmax', name='country_output')(lstm_layer)

# Define the model with three outputs
model = Model(inputs=input_layer, outputs=[stock_code_output, unit_price_output, country_output])

# Compile the model with different loss functions for each output
model.compile(
    loss={
        'stock_code_output': 'sparse_categorical_crossentropy',
        'unit_price_output': 'mse',
        'country_output': 'sparse_categorical_crossentropy'
    },
    optimizer='adam',
    metrics={
        'stock_code_output': 'accuracy',
        'unit_price_output': 'mae',
        'country_output': 'accuracy'
    }
)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X,
    {'stock_code_output': y_stock_code, 'unit_price_output': y_unit_price, 'country_output': y_country},
    epochs=50,
    batch_size=1000,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Save the model
model.save('text_model1.h5')
