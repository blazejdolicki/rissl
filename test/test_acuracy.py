import torch
import numpy as np

def test_accuracy(test_data, test_labels):
    correct = 0.0
    total = 0
    for idx, outputs in enumerate(test_data):
        batch_size = outputs.shape[0]
        total += batch_size
        correct += ((outputs > 0.0) == test_labels[idx]).sum().item()
    train_epoch_acc = 100 * correct / total
    return train_epoch_acc

if __name__ == "__main__":
    # Test case #1 - all outputs correct
    test_data = torch.Tensor([[0.55, -0.67, 0.08],
                             [-0.29, 0.71, -0.38]])
    test_labels = torch.Tensor([[1, 0, 1],
                                [0, 1, 0]])
    results = test_accuracy(test_data, test_labels)
    assert results == 100

    # Test case #2 - one output incorrect
    test_data = torch.Tensor([[0.55, -0.67, 0.08],
                              [-0.29, 0.71, -0.38]])
    test_labels = torch.Tensor([[1, 0, 1],
                                [1, 1, 0]])
    results = test_accuracy(test_data, test_labels)

    assert np.round(results, 2) == 83.33

    # Test case #2 - all outputs incorrect
    test_data = torch.Tensor([[-0.55, 0.67, -0.08],
                              [0.29, -0.71, 0.38]])
    test_labels = torch.Tensor([[1, 0, 1],
                                [0, 1, 0]])
    results = test_accuracy(test_data, test_labels)

    assert np.round(results, 2) == 0

    print("Tests passed")
