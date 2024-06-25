import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

class CNNService:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_indices = None  # Initialize class_indices

    def predict_product_class(self, image_file):
        image = tf.keras.preprocessing.image.load_img(image_file, target_size=(128, 128))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, axis=0)  # Expand dimensions to match the model input shape
        predictions = self.model.predict(image_array)
        class_index = tf.argmax(predictions[0]).numpy()
        class_name = list(self.class_indices.keys())[list(self.class_indices.values()).index(class_index)]
        return class_name

# Load training data
df = pd.read_csv('CNN_Model_Train_Data.csv')

# Ensure the CSV has 'image_path' and 'class' columns
if 'image_path' not in df.columns or 'class' not in df.columns:
    raise ValueError("CSV file must contain 'image_path' and 'class' columns")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    df,
    x_col='image_path',
    y_col='class',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    df,
    x_col='image_path',
    y_col='class',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Update class_indices in CNNService
CNNService.class_indices = train_generator.class_indices

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=5)

# Save the model
model.save('cnn_model1.h5')
