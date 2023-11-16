import functions as fn
import config



### DATA ###
## Train 
train_scenes, test_scenes = fn.load_dataframes(train_df, test_df)
normalized_train_scenes, normalized_test_scenes = fn.normalize_scenes(train_scenes, test_scenes, config.cols_to_normalize) # Call the normalization function
train_scenes_final = fn.split_scenes(train_scenes)
train_pcl = fn.process_scenes(train_scenes_final)
train_point_clouds, train_point_cloud_labels = fn.split_point_clouds(train_pcl)
plane_indices = fn.extract_plane_indices(train_point_clouds)

# Upsample minority class
train_point_clouds, train_label_cloud = fn.augment_additional_data(train_point_clouds, train_point_cloud_labels, plane_indices)
train_point_clouds, train_label_cloud = fn.shuffle_data(train_point_clouds, train_label_cloud)


## Test 
test_scenes_final = fn.split_scenes(test_scenes)
test_pcl = fn.process_scenes(test_scenes_final)
test_point_clouds, test_point_cloud_labels = fn.split_point_clouds(test_pcl)


### CREATE DATASETS ###
train_dataset = fn.generate_dataset(train_point_clouds, train_label_cloud)
val_dataset = fn.generate_dataset(test_point_clouds, test_point_cloud_labels, is_training=False)

### TRAINING POINTNET ###
lr_schedule, lrs = fn.create_lr_schedule(train_point_clouds, config.EPOCHS, config.BATCH_SIZE, config.INITIAL_LR)
segmentation_model, history = fn.run_experiment(epochs, lr_schedule, train_dataset, val_dataset)
