
# Build

docker build --network=host --tag rosbag2-language-sam .

# Use

docker run --net host -v $(pwd):/rosbag2_language_sam --gpus all -it rosbag2-language-sam

python main.py --input-path input_bag_folder --output-path output_bag_folder --config your_config.json