def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification
    :param prediction: np array of int (num_samples) - model predictions
    :param ground_truth: np array of int (num_samples) - true labels
    :return:
        accuracy - ratio of accurate predictions to total samples
    """
    num = ground_truth.shape[0]
    res = 0
    for i in range(num):
        if prediction[i] == ground_truth[i]:
            res += 1
    return res / num
