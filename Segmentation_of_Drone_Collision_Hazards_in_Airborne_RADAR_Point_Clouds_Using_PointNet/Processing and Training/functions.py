import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.utils

import config
import functions as fn
import model



#######################
### DATA PROCESSING ###
#######################

def load_dataframes(train_df, test_df):
    """
    Loads processed and labelled train and test CSV files into Pandas DataFrames.

    Returns:
    - tuple: A tuple containing three Pandas DataFrames corresponding to the files.
    """
    try:
        train_scenes = pd.read_csv(train_df)
        test_scenes = pd.read_csv(test_df)
    except FileNotFoundError as e:
        
        print(f"File not found: {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    
    return train_scenes, test_scenes




def normalize_scenes(train_scenes, test_scenes, cols_to_normalize):
    """
    Normalizes specified columns in the train and test scenes.

    Parameters:
    - train_scenes (list of DataFrame): List of DataFrames representing training scenes.
    - test_scenes (list of DataFrame): List of DataFrames representing testing scenes.
    - cols_to_normalize (list of str): Columns to normalize.

    Returns:
    - tuple: Tuple containing normalized train and test scenes.
    """
    # Concatenate all scenes for normalization
    all_scenes = pd.concat(train_scenes + test_scenes, ignore_index=True)

    # Calculate maximum absolute values for normalization
    max_abs_values = all_scenes[cols_to_normalize].apply(lambda x: np.max(np.abs(x)))

    # Normalize each scene
    normalized_train_scenes = [scene[cols_to_normalize].div(max_abs_values) for scene in train_scenes]
    normalized_test_scenes = [scene[cols_to_normalize].div(max_abs_values) for scene in test_scenes]

    return normalized_train_scenes, normalized_test_scenes




def split_scenes(scenes):
    """
    Splits each scene into smaller scenes based on a predefined timeframe.

    Parameters:
    - scenes (list of DataFrame): List of scenes (DataFrames) to be split.

    Returns:
    - list of DataFrame: List of split scenes.
    """
    scenes_final = []
    for scene in scenes:
        
        scene['timestamp'] = scene['Host Timestamp']

        start_time = scene['timestamp'].iloc[1]
        end_time = scene['timestamp'].iloc[-1]
        total_time = (end_time - start_time).total_seconds()

        number_of_scenes = int(total_time / config.TIMEFRAME)
        split_points = [start_time + pd.Timedelta(seconds=k * config.TIMEFRAME) for k in range(number_of_scenes + 1)]

        for start, end in zip(split_points[:-1], split_points[1:]):
            temp_scene = scene[(scene['timestamp'] > start) & (scene['timestamp'] <= end)]
            scenes_final.append(temp_scene.reset_index(drop=True))

    return scenes_final


def process_scenes(scenes):
    """
    Processes each scene by filtering out NaN values and filtering relevant columns.

    Parameters:
    - scenes (list of DataFrame): List of scenes to be processed.

    Returns:
    - list of DataFrame: List of processed scenes.
    """
    processed_pcl = []
    for scene in scenes:
        # Add 'Mask' column with default value 1
        scene['Mask'] = 1

        # Filter out scenes with NaN values
        if not scene.isnull().values.any():
            selected_columns = ['Date-Time', 'Xpos', 'Ypos', 'Zpos', 'Doppler', 'Power', 'Label', 'Mask']
            clean_pcl = scene[selected_columns].copy()
            processed_pcl.append(clean_pcl)

    return processed_pcl


def split_point_clouds(point_clouds):
    """
    Processes a list of point clouds by ensuring each has a fixed number of points and converts labels to one-hot encoding.

    Parameters:
    - point_clouds (list of DataFrame): List of point cloud DataFrames.

    Returns:
    - Tuple: Processed point clouds and their corresponding one-hot encoded labels.
    """
    processed_point_clouds = []
    point_cloud_labels = []

    for point_cloud in point_clouds:
        # Resize point clouds to have a fixed number of points
        if point_cloud.shape[0] >= config.NUM_SAMPLE_POINTS:
            resized_pcl = point_cloud.iloc[:config.NUM_SAMPLE_POINTS, :].to_numpy()
        else:
            repeat_factor = int(np.ceil(config.NUM_SAMPLE_POINTS / point_cloud.shape[0]))
            resized_pcl = np.tile(point_cloud.to_numpy(), (repeat_factor, 1))[:config.NUM_SAMPLE_POINTS]

        # Split features and labels
        features = resized_pcl[:, :5]
        labels = resized_pcl[:, 6]

        # Convert labels to one-hot encoding
        label_data = keras.utils.to_categorical(labels, config.NB_CLASSES)

        processed_point_clouds.append(features)
        point_cloud_labels.append(label_data)

    return processed_point_clouds, point_cloud_labels


def extract_plane_indices(point_clouds):
    """
    Extracts indices of point clouds where a specific label is present.

    Parameters:
    - point_clouds (list of DataFrame): List of point cloud DataFrames.

    Returns:
    - list: Indices of point clouds containing the specified label.
    """
    plane_indices = []
    for i, point_cloud in enumerate(point_clouds):
        # Check if label '3' exists in the 'Label' column
        if any(point_cloud['Label'] == 3):
            plane_indices.append(i)
    return plane_indices

def augment_additional_data(train_point_clouds, train_label_cloud, plane_indices, augmentation_factor=40):
    """
    Augments additional data for the training set based on specified indices.

    Parameters:
    - train_point_clouds (list): List of training point clouds.
    - train_label_cloud (list): List of training label clouds.
    - plane_indices (list): List of indices to be used for augmentation.
    - augmentation_factor (int): Factor by which to augment the data.

    Returns:
    - tuple: Tuple containing augmented training point clouds and labels.
    """
    additional_train_clouds = [train_point_clouds[i] for i in plane_indices] * augmentation_factor
    additional_train_labels = [train_label_cloud[i] for i in plane_indices] * augmentation_factor

    train_point_clouds += additional_train_clouds
    train_label_cloud += additional_train_labels

    return train_point_clouds, train_label_cloud

def shuffle_data(point_clouds, label_clouds):
    """
    Shuffles point clouds and their corresponding labels.

    Parameters:
    - point_clouds (list): List of point clouds.
    - label_clouds (list): List of label clouds.

    Returns:
    - tuple: Tuple containing shuffled point clouds and labels.
    """
    combined = list(zip(point_clouds, label_clouds))
    random.shuffle(combined)
    shuffled_point_clouds, shuffled_label_clouds = zip(*combined)
    return list(shuffled_point_clouds), list(shuffled_label_clouds)

def generate_dataset(point_clouds, label_clouds, is_training=True):
    """
    Generates a TensorFlow dataset for training or validation.

    Parameters:
    - point_clouds (list): List of point clouds.
    - label_clouds (list): List of label clouds.
    - is_training (bool): Flag to indicate if the dataset is for training.

    Returns:
    - tf.data.Dataset: The generated TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((point_clouds, label_clouds))
    dataset = dataset.shuffle(BATCH_SIZE * 100) if is_training else dataset
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset




################
### TRAINING ###
################

def create_lr_schedule(train_point_clouds, epochs, batch_size, initial_lr, schedule_boundaries=[1, 3]):
    """
    Creates a learning rate schedule for training.

    Parameters:
    - train_point_clouds (list): List of training point clouds.
    - epochs (int): Number of epochs for training.
    - batch_size (int): Batch size for training.
    - initial_lr (float): Initial learning rate.
    - schedule_boundaries (list): List of epochs at which the learning rate changes.

    Returns:
    - tf.keras.optimizers.schedules.LearningRateSchedule: The learning rate schedule.
    - list: List of learning rates for each training step.
    """
    training_step_size = len(train_point_clouds) // batch_size
    total_training_steps = training_step_size * epochs
    print(f"Total training steps: {total_training_steps}.")

    boundaries = [training_step_size * boundary for boundary in schedule_boundaries]
    values = [initial_lr] + [initial_lr / (2 ** i) for i in range(1, len(schedule_boundaries) + 1)]

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries,
        values=values,
    )

    steps = tf.range(total_training_steps, dtype=tf.int32)
    lrs = [lr_schedule(step).numpy() for step in steps]

    return lr_schedule, lrs



def run_experiment(epochs, lr_schedule, train_dataset, val_dataset):

    segmentation_model = model.get_shape_segmentation_model(config.NUM_SAMPLE_POINTS, config.NB_CLASSES)
    segmentation_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    checkpoint_filepath = config.checkpoint_filepath
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )
    

    history = segmentation_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        #sample_weight = {0:1,1:1,2:10,3:1000},
        callbacks=[checkpoint_callback],
    )

    segmentation_model.load_weights(checkpoint_filepath)
    return segmentation_model, history