# Image_Classification
This project is about building an image classification model that can accurately recognize and categorize different types of flowers from images. For this task, I chose to work with a flower dataset that contains multiple categories like roses, sunflowers, tulips, dandelions, and daisies. Each image represents one class, and the objective is to train a model that can correctly classify unseen flower images into their respective categories.

**About Dataset:**

The flower dataset contains images organized into different folders, each representing a category. For example, images of roses are stored under the folder named "roses", and so on for other flowers. I used the `pathlib` and `tensorflow.keras.utils.image_dataset_from_directory()` to load and process the image data. The dataset is structured such that images are automatically labeled based on the folder name, which makes it easy to use for supervised learning.

**About Environment:**

I used Jupyter Notebook for this project, as it allows me to write and test code in cells, visualize images directly, and view training metrics in real-time. I used TensorFlow and Keras libraries for building and training the CNN model. I also used NumPy, Matplotlib, and Scikit-learn for additional processing and evaluation.

#### Steps Involved:

**Step 1:** Data Loading and Preprocessing
I started by loading the dataset using image_dataset_from_directory, resizing the images to a fixed shape (e.g., 180x180), and scaling the pixel values to the range [0,1] by dividing by 255. I also split the data into training and testing sets to evaluate model performance later.

**Step 2:** Visualizing the Dataset
To get an idea of the image content, I visualized sample images from each class. This helps to ensure that the dataset is correctly loaded and balanced across categories.

**Step 3:** Building the CNN Model
I used Keras’ Sequential API to define a Convolutional Neural Network. The model includes multiple Conv2D and MaxPooling2D layers to extract features, followed by Flatten and Dense layers to classify the images. I used softmax activation in the output layer since it’s a multi-class classification problem.

**Step 4:** Model Compilation and Training

I compiled the model using Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric. Then, I trained the model for a few epochs (e.g., 10) using the training data and validated it on the test set.

**Step 5:** Model Evaluation

After training, I evaluated the model on the test data. I used classification_report and confusion_matrix from Scikit-learn to get detailed performance metrics like precision, recall, f1-score, and class-wise accuracy.

**Step 6:** Making Predictions

I made predictions on individual test images, displayed the image, and printed both the true label and the predicted label. This helped in manually checking how well the model performed visually.

## Conclusion:

This image classification task gave me practical exposure to handling image datasets, building CNN models, and evaluating their performance. It also taught me how to visualize results and identify areas for improvement. Overall, this was an exciting hands-on project combining computer vision and deep learning concepts.
