""" Configuration file for the Machine learning application. """

LOADER_NAME = "local_data_loader"

# Local Training data file path
TRAINING_DATA_FILE = "../data/raw/mnist_train.csv"
TEST_DATA_FILE = "../data/raw/mnist_test.csv"

# Output file paths
PREDICTIONS_PATH = "../results/predictions/"
FIGURE_PATH = "../results/figures/"
REPORT_PATH = "../results/reports/"

# Target features
TARGET_FEATURE = "label"
