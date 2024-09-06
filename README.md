# U-2-Net-Computer-Vision-Project
U-2-Net Computer Vision Project: Image Segmentation, Car Orientation Classification, and Virtual Background Augmentation

Task 1:
1. Imports:

•	os: Used for file system operations like listing files in a directory.

•	numpy (np): Used for numerical computations (not directly used in this code).

•	torch: The PyTorch library for deep learning.

•	torch.nn: Submodule of PyTorch containing building blocks for neural networks.

•	torch.optim: Submodule of PyTorch containing optimizers for training neural networks.

•	from torch.utils.data import Dataset, DataLoader: Provides tools for creating and managing datasets and data loaders.

•	transforms: Submodule of torchvision for image transformations.

•	PIL.Image: Python Imaging Library for loading and manipulating images.

2. Dataset Paths:

•	image_dir: Defines the path to the directory containing car images.

•	mask_dir: Defines the path to the directory containing masks for the car images.

3. Data Augmentation and Preprocessing:

•	image_transform: Defines a series of transformations for image data:

o	transforms.Resize((320, 320)): Resizes images to a fixed size (320x320 pixels) to ensure compatibility with the model.

o	transforms.RandomHorizontalFlip(): Randomly flips images horizontally with a 50% chance. This helps the model learn to be robust to different object orientations.

o	transforms.RandomVerticalFlip(): Randomly flips images vertically with a 50% chance, similar to horizontal flip.

o	transforms.ToTensor(): Converts PIL Images to PyTorch tensors.

o	transforms.Normalize(): Normalizes the pixel values to a range between 0 and 1. This helps improve model convergence. The specific mean and standard deviation values used here are common for images in the ImageNet dataset.

•	mask_transform: Defines transformations for mask data:

o	transforms.Resize((320, 320)): Same as for images, ensures consistent size.

o	transforms.RandomHorizontalFlip(): Same as for images.

o	transforms.RandomVerticalFlip(): Same as for images.

o	transforms.ToTensor(): Converts PIL Images to PyTorch tensors.

o	(No normalization): Masks are typically binary (0 or 1) representing object presence, so normalization isn't necessary.

4. CarDataset Class:

•	This class represents the dataset containing the car images and masks.

•	init: Initializes the class with image paths, mask paths, and the defined transforms for images and masks.

•	len: Returns the length of the dataset (number of image-mask pairs).

•	getitem: Retrieves a specific data point (image and mask) at the given index. It loads the image and mask from their paths, applies the corresponding transforms, and returns the processed image and mask. The mask is also reshaped to have an extra dimension (batch dimension of 1) to match the expected input format for the model.

5. Dataset Loading:

•	image_files: Lists all file paths within the image directory.

•	mask_files: Lists all file paths within the mask directory.

•	train_test_split: Splits the dataset into training and validation sets using scikit-learn's train_test_split. This ensures the model is trained on a subset of the data and evaluated on another unseen subset. The test size is set to 0.25, meaning 25% of the data will be used for validation.

o	train_image_files, val_image_files: Lists of image paths for training and validation sets.

o	train_mask_files, val_mask_files: Lists of mask paths for training and validation sets.

•	CarDataset instances: Creates instances of the CarDataset class for the training and validation sets, using the respective image and mask paths and the defined transforms.

•	Dataloaders: Creates dataloaders using DataLoader to manage the datasets during training and validation. Dataloaders efficiently batch the data for processing by the model.

6. Device Selection:

•	device: Checks if a GPU is available and sets the device to "cuda:0" (first GPU) if so, otherwise it uses the CPU ("cpu"). Training on a GPU is significantly faster than CPU if available.

7. U-2-Net Model:

•	U2Net class: Defines the U-2-Net neural network architecture.

o	encoder: A series of convolutional layers with ReLU activations and max pooling to extract features from the image. Padding is added in the convolutional layers to preserve spatial

**Task 2:**

1. Imports:

•	torch, torch.nn, torch.optim, from torch.utils.data import Dataset, DataLoader: Same as the previous code, used for deep learning model building, data management, and training.

•	torchvision.transforms as transforms: Imports image transformation functions from torchvision.

•	PIL.Image: Used for loading and manipulating images.

•	numpy (np): Used for numerical computations (not directly used in this code for computations).

•	cv2 (OpenCV): Not directly used in this code, but could potentially be used for image processing tasks.

•	glob: Used to find all files matching a specific pattern in a directory.

•	os: Used for file system operations.

•	random: Used for generating random numbers for data splitting.

2. Data Transforms:

•	transform_image: Defines a series of transformations for image data:

o	transforms.Resize((320, 320)): Resizes images to a fixed size (320x320) for consistency with the model.

o	transforms.ToTensor(): Converts PIL Images to PyTorch tensors.

o	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]): Normalizes pixel values to a range between 0 and 1 for better model convergence. These values are commonly used for images normalized in the range [0, 1].

•	transform_mask: Defines transformations for mask data:

o	transforms.Resize((320, 320)): Resizes masks to match the image size.

o	transforms.ToTensor(): Converts PIL Images to PyTorch tensors.

o	(No normalization): Masks are typically binary (0 or 1) representing object presence, so normalization isn't necessary.

3. Dataset Loading:

•	image_folder, mask_folder: Define the paths to directories containing car images and masks.

•	image_paths, mask_paths: Use glob to find all image and mask files within their respective folders.

•	Splitting data:

o	train_size: Defines the size of the training set (80% of the data).

o	val_size: Calculated as the remaining 20% of the data.

o	random.sample: Randomly samples train_size indices to create the training set.

o	train/val indices: Separate lists of indices for training and validation data.

o	train/val_image/mask_paths: Separate lists of image and mask paths for training and validation sets based on the chosen indices.

4. CarDataset Class:

•	This class represents the custom dataset containing car images and masks.

o	init: Initializes the class with image and mask paths, and optionally takes transform and target_transform functions for additional processing.

o	len: Returns the length of the dataset (number of image-mask pairs).

o	getitem: Retrieves a specific data point (image and mask) at the given index. It loads the image and mask from their paths, applies the corresponding transforms (if provided), and returns the processed image and mask. The mask is reshaped to have an extra dimension (batch dimension of 1) to match the expected input format for the model. The target_transform allows for applying different transformations to images and masks if needed.

5. Data Loaders:

•	train_dataset, val_dataset: Create instances of the CarDataset class for training and validation sets, using the respective image and mask paths and the defined transforms.

•	train_loader, val_loader: Create dataloaders for training and validation using DataLoader to manage the datasets during training and validation. Dataloaders efficiently batch the data for processing by the model. Here, a batch size of 32 is used.

6. U-Net Model:

•	U2Net class: Defines the U-Net neural network architecture.

o	This is a commonly used architecture for semantic segmentation tasks like this one, where the goal is to predict a mask for each pixel in the image.

o	It uses a series of convolutional layers with ReLU activations for feature extraction, followed by upsampling layers to recover spatial information and predict the segmentation mask.

7. Model Training:

•	device: Checks if a GPU is available and sets the device to use for training (GPU or CPU).

•	model.to(device): Moves the model to the chosen device (GPU or CPU).

**Task 3:**

1. Imports:

•	cv2: Open Computer Vision library, used for image processing tasks.

•	glob: Used to find all files matching a specific pattern in a directory.

•	os: Used for file system operations.

2. Path Definition:

•	image_folder, mask_folder: Define the paths to directories containing car images and masks.

3. Image and Mask Paths:

•	image_paths, mask_paths: Use glob to find all image and mask files within their respective folders. The sorted function ensures the paths are in a consistent order.

4. get_orientation Function:

•	Purpose: Determines the orientation (horizontal or vertical) of the car in a given mask image.

•	Steps:

1.	Load the mask: Loads the mask image in grayscale mode using cv2.imread.

2.	Find contours: Uses cv2.findContours to detect the outlines of objects (in this case, the car) within the mask.

3.	Check for contours: If no contours are found, it returns "No car detected" as the orientation.

4.	Get largest contour: Finds the largest contour, assuming it corresponds to the car.

5.	Calculate bounding box: Obtains the bounding rectangle for the largest contour using cv2.boundingRect.

6.	Determine orientation: Calculates the aspect ratio (width/height) of the bounding box. If the width is greater than the height, the orientation is considered "Horizontal"; otherwise, it's considered "Vertical".

5. Testing:

•	Iterates over the first 5 image and mask paths.

•	Calls the get_orientation function for each mask path and prints the image filename and its corresponding orientation.

**Task 4:**

1.	Image and Mask Loading:

Purpose: Loads the car images and their corresponding masks from specified paths.

Reasoning: This step is essential for obtaining the input data that will be processed to add the virtual background and shadow.

2.	Background Creation:

Purpose: Creates a virtual background of the desired type (white or gradient).

Reasoning: The background provides the context for placing the car. A white background is a common choice, but a gradient can add visual interest.

3.	Mask Handling:

Purpose: Extracts the car region from the image using the mask and isolates the background region.

Reasoning: The mask acts as a guide to separate the car from its surroundings. By isolating the car and background regions, we can manipulate them independently.

4.	Blending:

Purpose: Combines the car region with the virtual background to create the final image.

Reasoning: The cv2.add function is used to merge the car and background regions, ensuring that the car is placed on top of the background.

5.	Shadow Simulation:

Purpose: Simulates a realistic shadow beneath the car.

Reasoning: A shadow adds depth and realism to the scene. The shadow is created by dilating and blurring the mask, then applying a darkening effect.

6.	Shadow Offset:

Purpose: Adjusts the vertical position of the shadow.

Reasoning: The shadow_offset parameter allows you to control the distance between the car and the shadow. A small offset typically looks more natural.

7.	Output:

Purpose: Saves the final image with the added background and shadow.

Reasoning: The cv2.imwrite function is used to save the processed image to the specified output path.
