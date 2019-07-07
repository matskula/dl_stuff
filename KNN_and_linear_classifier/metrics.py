def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(prediction)):
        if prediction[i] and ground_truth[i]:
            tp += 1
        elif prediction[i] and not ground_truth[i]:
            fp += 1
        elif not prediction[i] and ground_truth[i]:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / len(prediction)
    f1 = 2 * recall * precision / (recall + precision)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    num = ground_truth.shape[0]
    res = 0
    for i in range(num):
        if prediction[i] == ground_truth[i]:
            res += 1
    return res / num
