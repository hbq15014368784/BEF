import torch

def load_features(feature_file_path):
    # Load the features from the .pth file
    features = torch.load(feature_file_path, map_location=torch.device('cpu'))
    return features

if __name__ == "__main__":

    feature_file_path = 'data_RAD/rcnn_feature/synpic100132.pth'
    features = load_features(feature_file_path)
    print(features['pool'].shape)