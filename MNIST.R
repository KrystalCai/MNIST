# ---------------------------
# Introduction
# ---------------------------
# The MNIST dataset is a widely used benchmark for handwritten digit recognition, 
# containing 70,000 grayscale images of digits from 0 to 9.
# This project focuses on building a convolutional neural network (CNN) to classify these digits accurately.
# Data preprocessing and augmentation techniques are applied to improve model generalization.
# The pipeline includes model training, evaluation using precision, recall, and F1 score, 
# as well as optimization to enhance performance.
# The goal is to demonstrate a complete machine learning workflow for image classification tasks.


library(keras)
library(reticulate)
library(tensorflow)
# Step 1: Load the MNIST dataset
# Load dataset from keras package
mnist <- dataset_mnist()

# Get the training and test images and labels
# Training set images and labels
train_images <- mnist$train$x
train_labels <- mnist$train$y

# Test set images and labels
test_images <- mnist$test$x
test_labels <- mnist$test$y

# Normalize the pixel values to the range [0, 1]
# This helps to speed up model convergence
train_images <- train_images / 255
test_images <- test_images / 255

# Reshape the data to fit the input format of the neural network
# The shape should be [samples, 28, 28, 1] for grayscale images
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# One-hot encode the training labels (convert the labels to categorical format)
train_labels <- to_categorical(as.vector(mnist$train$y), num_classes = 10)
# One-hot encode the test labels (convert the labels to categorical format)
test_labels <- to_categorical(as.vector(mnist$test$y), num_classes = 10)

# Check the shape of the data (e.g., number of samples and image dimensions)
dim(train_images)
dim(test_images)


# Step 2: Build a Convolutional Neural Network (CNN)

model <- keras_model_sequential() %>%
  # Add the first convolutional layer
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  # Add a max pooling layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Add the second convolutional layer
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  # Add another max pooling layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Add a third convolutional layer
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  
  # Flatten the output of the convolutions to feed into the fully connected layers
  layer_flatten() %>%
  
  # Add a fully connected (dense) layer
  layer_dense(units = 64, activation = 'relu') %>%
  
  # Output layer with 10 units (one for each digit)
  layer_dense(units = 10, activation = 'softmax')  # Use softmax for multi-class classification


# Step 3: Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',    # Loss function for multi-class classification
  optimizer = optimizer_adam(),         # Optimizer for training
  metrics = c('accuracy')               # Evaluate the model based on accuracy
)

# Step 4: Train the model
history <- model %>% fit(
  train_images,                # Training images
  train_labels,                # Training labels
  epochs = 5,                  # Number of training epochs
  batch_size = 64,             # Batch size
  validation_data = list(test_images, test_labels)  # Validation set for evaluation
)


# Step 5: Evaluate the model on the test set
score <- model %>% evaluate(test_images, test_labels)
# Print the test loss and accuracy
cat('Test loss:', score[1], '\n') #Test loss: 0.02575469 
cat('Test accuracy:', score[2], '\n') #Test accuracy: 0.9919 

# Step 6: Make predictions on the test set
predictions <- model %>% predict(test_images)

# Convert the predictions to class labels
predicted_labels <- apply(predictions, 1, which.max) - 1  # Subtract 1 to get 0-based index

# Show the first 10 test images with their predicted labels and true labels
par(mfrow = c(2, 5), mar = c(1,1,1,1))  # Create a 2x5 grid to display images
for (i in 1:10) {
  image(test_images[i,, ,1], col = grey.colors(256), axes = FALSE)  # Plot the image
  title(paste("Predicted:", predicted_labels[i], "True:", which.max(test_labels[i,])-1))  # Add title with predicted and true labels
}


# Create a confusion matrix
table(Predicted = predicted_labels, Actual = apply(test_labels, 1, which.max)-1)

# Save the trained model to a file
save_model_hdf5(model, "mnist_model.h5")

# Load the saved model
#model <- load_model_hdf5("mnist_model.h5")
model %>% save_model_tf("mnist_model.keras")


# Load the saved model
model_loaded <- load_model_tf("mnist_model.keras")



# Verify that the model works by predicting
predictions_loaded <- model_loaded %>% predict(test_images)


# Show the first 10 predicted labels vs. true labels
predicted_labels_loaded <- apply(predictions_loaded, 1, which.max) - 1  # Convert one-hot encoding back to labels
true_labels <- apply(test_labels, 1, which.max) - 1  # Convert one-hot encoding to true labels

# Display the comparison
data.frame(Predicted = predicted_labels_loaded[1:10], Actual = true_labels[1:10])

# Show the first 10 test images with their predicted labels and true labels
par(mfrow = c(2, 5), mar = c(1, 1, 1, 1))  # Create a 2x5 grid to display images
for (i in 1:10) {
  image(test_images[i,, ,1], col = grey.colors(256), axes = FALSE)  # Plot the image
  title(paste("Predicted:", predicted_labels_loaded[i], "True:", true_labels[i]))  # Add title with predicted and true labels
}


# Create confusion matrix for the loaded model
conf_matrix <- table(Predicted = predicted_labels_loaded, Actual = true_labels)
conf_matrix

#----------------------------
## Hyperparameter Tuning and Model Optimization
#----------------------------
# Try different optimizers and adjust learning rates to improve model performance
# Using Adam optimizer here, but you can also try SGD, RMSprop, etc.
model %>% compile(
  loss = 'categorical_crossentropy',  # Loss function for multi-class classification
  optimizer = optimizer_adam(learning_rate = 0.001),  # You can experiment with different learning rates
  metrics = c('accuracy')  # Track accuracy during training
)

# Train the model with different batch sizes and epochs to improve performance
history <- model %>% fit(
  train_images,                # Training images
  train_labels,                # Training labels
  epochs = 10,                 # Increase epochs for better training (you can try more)
  batch_size = 32,             # Smaller batch size (adjustable)
  validation_data = list(test_images, test_labels)  # Validation set for evaluation
)



#-------------------------
#Add Regularization and Prevent Overfitting
#--------------------------
# Add Dropout layers to the model to prevent overfitting
# Dropout randomly sets a fraction of input units to 0 at each update during training
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%  # Adding Dropout layer to prevent overfitting
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%  # Adding Dropout layer to prevent overfitting
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%  # Adding Dropout layer to prevent overfitting
  layer_dense(units = 10, activation = 'softmax')  # Final output layer

# Compile and train the model with the updated structure
model %>% compile(
  loss = 'categorical_crossentropy',  # Multi-class classification loss
  optimizer = optimizer_adam(),       # Adam optimizer
  metrics = c('accuracy')             # Track accuracy during training
)

# Train the updated model
history <- model %>% fit(
  train_images, train_labels,
  epochs = 10, batch_size = 32,
  validation_data = list(test_images, test_labels)
)


#-------------------------------
#Data Augmentation
#--------------------------------
library(keras)
library(tensorflow)
library(dplyr)



#-------------------------
# Evaluate Additional Metrics (Precision, Recall, F1 Score)
#----------------------------
#install.packages("yardstick")
library(yardstick)

# Convert predictions and true labels into factors for evaluation
#predicted_labels <- factor(predicted_labels, levels = 0:9)
truth <- apply(test_labels, 1, which.max) - 1
predicted_labels <- apply(predictions, 1, which.max) - 1
truth <- factor(truth, levels = 0:9)
predicted_labels <- factor(predicted_labels, levels = 0:9)

results <- data.frame(
  truth = truth,
  .pred_class = predicted_labels
)


# Compute Precision, Recall, and F1 Score for each class
precision_score <- precision(results, truth = truth, estimate = .pred_class)
recall_score    <- recall(results, truth = truth, estimate = .pred_class)
f1_score        <- f_meas(results, truth = truth, estimate = .pred_class)


# Print evaluation metrics
precision_score
recall_score
f1_score


#table(results$truth)
#table(results$.pred_class)

# Convert true labels and predicted labels to factors with levels 0-9
truth <- factor(truth, levels = 0:9)# True labels (ground truth)
predicted_labels <- factor(predicted_labels, levels = 0:9)# Predicted labels from the model
# Extract all class labels
classes <- levels(truth)
# Initialize a data frame to store precision, recall, and F1 score for each class
metrics_by_class <- data.frame(
  class = classes,# Digit classes 0-9
  precision = NA,# Placeholder for precision
  recall = NA,# Placeholder for recall
  f1 = NA# Placeholder for F1 score
)

# Loop over each class to calculate per-class metrics using a one-vs-all approach
for (i in seq_along(classes)) {
  cls <- classes[i]
  
  # Convert true and predicted labels into binary: current class vs. all others
  truth_bin <- factor(ifelse(truth == cls, cls, paste0("not_", cls)))# Binary truth
  pred_bin  <- factor(ifelse(predicted_labels == cls, cls, paste0("not_", cls)))# Binary prediction
  
  # Compute precision, recall, and F1 for the current class
  metrics_by_class$precision[i] <- precision_vec(truth_bin, pred_bin, event_level = "first")
  metrics_by_class$recall[i]    <- recall_vec(truth_bin, pred_bin, event_level = "first")
  metrics_by_class$f1[i]        <- f_meas_vec(truth_bin, pred_bin, event_level = "first")
}

# Display the per-class metrics
metrics_by_class

#---------------------------------------------
##############################################
#Model Classification Summary:
##############################################
#---------------------------------------------
#Overall performance is very strong, with F1 scores above 0.98 for all 10 classes.

#Precision is high across the board, indicating very few false positives.

#Recall is also high for most classes, though class 9 is slightly lower (0.974), suggesting a few missed samples.

#The model is stable for the majority of classes, with minor room for improvement in a few cases.



library(ggplot2)
library(caret)
library(reshape2)
# -----------------------------
# 1. Confusion Matrix Visualization
# -----------------------------
# Predict class for test set
predicted_classes <- apply(predictions, 1, which.max) - 1
true_classes <- apply(test_labels, 1, which.max) - 1

# Create confusion matrix
cm <- table(True = true_classes, Predicted = predicted_classes)
cm_df <- as.data.frame(cm)
colnames(cm_df) <- c("True", "Predicted", "Freq")

# Plot heatmap
ggplot(cm_df, aes(x = Predicted, y = True, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 3) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Confusion Matrix", x = "Predicted Class", y = "True Class")



# -----------------------------
# 2. Model Optimization Example
# -----------------------------
# For illustration: change optimizer to 'adam' and add dropout
library(keras)

# define a slightly improved CNN model
model_opt <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(10, activation = 'softmax')


# Compile model with optimized settings
model_opt %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# Train model (example: fewer epochs for testing)
history_opt <- model_opt %>% fit(
  train_images, train_labels,
  batch_size = 32,
  epochs = 5,
  validation_data = list(test_images, test_labels)
)



# -----------------------------
# 3. Evaluate optimized model
# -----------------------------
score <- model_opt %>% evaluate(test_images, test_labels)
cat("Test loss:", score[1], "\n")
cat("Test accuracy:", score[2], "\n")


# ---------------------------
# Summary
# ---------------------------
# This project builds and evaluates a classification model for handwritten digit recognition using the MNIST dataset. 
# The model achieved excellent performance on the test set, with a test loss of 0.0284 and an accuracy of 0.9915. 
# Class-wise evaluation metrics were computed, including precision, recall, and F1 score for each digit. 
# The results indicate that the model performs consistently well across most classes, with precision, recall, 
# and F1 scores all above 0.97 for all digits. These metrics demonstrate that the model not only predicts the correct digit 
# with high overall accuracy but also maintains strong per-class performance, which is crucial for multi-class classification tasks.
