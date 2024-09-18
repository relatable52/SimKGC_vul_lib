import torch

def accuracy_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int = 2) -> float:
    """Compute the top-k accuracy for a batch of logits and labels.
    
    Parameters
    ----------
    logits : torch.Tensor
        A tensor of shape (batch_size, num_classes) containing the predicted logits for each class.
    labels : torch.Tensor
        A tensor of shape (batch_size,) containing the true labels (as class indices).
    k : int, optional
        The number of top predictions to consider for accuracy, by default 5.
    
    Returns
    -------
    float
        The top-k accuracy, represented as a percentage of correctly classified samples.
    
    Raises
    ------
    ValueError
        If `k` is greater than the number of classes in `logits`.
    
    Examples
    --------
    >>> batch_size = 32
    >>> num_classes = 10
    >>> logits = torch.randn(batch_size, num_classes)  # Example logits
    >>> labels = torch.randint(0, num_classes, (batch_size,))  # Random true labels
    >>> accuracy_at_k(logits, labels, k=5)
    85.0
    """
    
    if k > logits.size(1):
        raise ValueError(f"k ({k}) cannot be greater than the number of classes ({logits.size(1)})")
    
    top_k_preds = torch.topk(logits, k=k, dim=1).indices  
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds)) 
    correct_count = correct.sum().item()
    accuracy = correct_count / logits.size(0) * 100 
    return accuracy