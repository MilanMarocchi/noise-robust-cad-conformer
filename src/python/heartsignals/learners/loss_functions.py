import torch.nn as nn
import torch
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F


class HFContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output):
        loss = output.loss

        return loss

class WrappedCrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, classifications, labels):
        return self.criterion(classifications, labels)

class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_c=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c  # Weight for center loss
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feature_dim))  # Learnable centers

    def forward(self, features, labels):
        batch_size = features.shape[0]
        centers_batch = self.centers[labels]  # Select centers corresponding to labels
        center_loss = ((features - centers_batch) ** 2).sum(dim=1).mean()  # L2 distance
        return self.lambda_c * center_loss

def contrastive_l2_loss_with_labels(features, labels, margin=1.0):
    """
    Compute contrastive loss using labels to define positive and negative pairs.

    Args:
        features (torch.Tensor): Tensor of shape (batch_size, feature_dim) containing embeddings.
        labels (torch.Tensor): Tensor of shape (batch_size,) containing class labels.
        temperature (float): Temperature parameter for scaling similarity scores.

    Returns:
        torch.Tensor: Scalar contrastive loss.
    """
    # Compute pairwise L2 distances (squared for numerical stability)
    distances = torch.cdist(features, features, p=2)  # L2 norm
    
    # Create positive and negative masks
    labels_expanded = labels.unsqueeze(0)
    positive_mask = labels_expanded == labels_expanded.t()  # Same class
    negative_mask = ~positive_mask  # Different class
    
    # Calculate positive loss (pull closer)
    positive_loss = torch.sum(distances * positive_mask.float()) / (positive_mask.sum() + 1e-8)
    
    # Calculate negative loss (push apart)
    negative_loss = torch.sum(F.relu(margin - distances) * negative_mask.float()) / (negative_mask.sum() + 1e-8)
    
    # Total contrastive loss
    loss = positive_loss + negative_loss
    return loss

def contrastive_loss_with_labels(features, labels, temperature=0.5):
    """
    Compute contrastive loss using labels to define positive and negative pairs.

    Args:
        features (torch.Tensor): Tensor of shape (batch_size, feature_dim) containing embeddings.
        labels (torch.Tensor): Tensor of shape (batch_size,) containing class labels.
        temperature (float): Temperature parameter for scaling similarity scores.

    Returns:
        torch.Tensor: Scalar contrastive loss.
    """
    batch_size = features.shape[0]

    # Normalize embeddings
    features = F.normalize(features, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(features, features.T)  # (batch_size, batch_size)

    # Create positive mask where labels match
    labels = labels.view(-1, 1)  # Reshape to (batch_size, 1)
    positive_mask = (labels == labels.T).float()  # Binary mask where labels match

    # Remove self-comparisons by setting diagonal to 0
    positive_mask.fill_diagonal_(0)

    # Apply temperature scaling
    logits = similarity_matrix / temperature

    # Mask out self-similarity (diagonal)
    logits = logits - torch.eye(batch_size, device=features.device) * 1e9

    # Compute log-softmax over all similarities
    log_probs = F.log_softmax(logits, dim=1)

    # Compute loss by focusing only on positive pairs
    loss = - (positive_mask * log_probs).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
    loss = loss.mean()

    return loss

class ContrastiveClassificationCenterLoss(nn.Module):
    def __init__(self, center_loss_fn=None, alpha=0.5, beta=0.4, temperature=0.7):
        super().__init__()
        self.center_loss_fn = center_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, features, classification, labels):
        # Compute contrastive loss
        contrastive_loss_value = contrastive_loss_with_labels(features, labels, self.temperature)

        # Compute classification loss
        logits_cls = classification
        classification_loss_value = F.cross_entropy(logits_cls, labels)

        # Compute center loss (only if provided)
        if self.center_loss_fn is not None:
            center_loss_value = self.center_loss_fn(features, labels)
        else:
            center_loss_value = torch.tensor(0.0, device=features.device)

        # Final loss (simplified)
        total_loss = self.beta * contrastive_loss_value + self.alpha * classification_loss_value + 0.01 * center_loss_value

        # Return total loss + breakdown
        loss_dict = {
            "total_loss": total_loss.item(),
            "contrastive_loss": contrastive_loss_value.item(),
            "classification_loss": classification_loss_value.item(),
            "center_loss": center_loss_value.item(),
        }

        return total_loss

class HFContrastiveClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, output_classify, labels):

        ss_loss = output.loss 
        classify_loss = 1000 * (F.cross_entropy(output_classify, labels))
        print(f"{ss_loss=}, {classify_loss=}")
        loss = ss_loss + classify_loss

        return loss

class ContrastiveHybridLoss(nn.Module):
    def __init__(self, 
                 center_loss_fn=None, 
                 alpha=0.5, 
                 beta=0.2, 
                 margin=2.0,
    ):
        super().__init__()
        self.center_loss_fn = center_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, features, classification, labels):
        # Compute contrastive loss
        contrastive_loss_value = contrastive_l2_loss_with_labels(features, labels, self.margin)

        # Compute classification loss
        logits_cls = classification
        classification_loss_value = F.cross_entropy(logits_cls, labels)

        # Compute center loss (only if provided)
        if self.center_loss_fn is not None:
            center_loss_value = self.center_loss_fn(features, labels)
        else:
            center_loss_value = torch.tensor(0.0, device=features.device)

        # Final loss (simplified)
        total_loss = self.beta * contrastive_loss_value + self.alpha * classification_loss_value + 0.01 * center_loss_value

        # Return total loss + breakdown
        loss_dict = {
            "total_loss": total_loss.item(),
            "contrastive_loss": contrastive_loss_value.item(),
            "classification_loss": classification_loss_value.item(),
            "center_loss": center_loss_value.item(),
        }

        return total_loss

class ContrastiveFocalLoss(nn.Module):
    def __init__(self, 
                 center_loss_fn=None, 
                 alpha=0.5, 
                 beta=0.2, 
                 temperature=0.7,
                 focal_alpha=1,
                 center=0.01,
                 focal_gamma=2,
                 focal_reduction='mean'
    ):
        super().__init__()
        self.center_loss_fn = center_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.center = center
        self.temperature = temperature

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_reduction = focal_reduction

    def focal_loss(self, classification, labels, reduction='mean'):
        BCE_loss = F.cross_entropy(classification, labels, reduction='none')
        
        # Compute the probabilities
        pt = torch.exp(-BCE_loss)
        
        # Compute the focal loss
        F_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * BCE_loss
        
        # Apply the reduction method
        if reduction == 'mean':
            return F_loss.mean()
        elif reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

    def forward(self, features, classification, labels):
        # Compute contrastive loss
        contrastive_loss_value = contrastive_loss_with_labels(features, labels, self.temperature)

        # Compute classification loss
        logits_cls = classification
        classification_loss_value = F.cross_entropy(logits_cls, labels)

        # Compute center loss (only if provided)
        if self.center_loss_fn is not None:
            center_loss_value = self.center_loss_fn(features, labels)
        else:
            center_loss_value = torch.tensor(0.0, device=features.device)

        # Final loss (simplified)
        total_loss = self.beta * contrastive_loss_value + self.alpha * classification_loss_value + self.center * center_loss_value

        # Return total loss + breakdown
        loss_dict = {
            "total_loss": total_loss.item(),
            "contrastive_loss": contrastive_loss_value.item(),
            "classification_loss": classification_loss_value.item(),
            "center_loss": center_loss_value.item(),
        }

        return total_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the cross entropy loss
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute the probabilities
        pt = torch.exp(-BCE_loss)
        
        # Compute the focal loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        # Apply the reduction method
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_c=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c  # Weight for center loss
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feature_dim))  # Learnable centers

    def forward(self, features, labels):
        batch_size = features.shape[0]
        centers_batch = self.centers[labels]  # Select centers corresponding to labels
        center_loss = ((features - centers_batch) ** 2).sum(dim=1).mean()  # L2 distance
        return self.lambda_c * center_loss

def orthogonality_loss(features):
    """
    Encourages diversity in features by penalizing similarity between embeddings.

    Args:
        features (torch.Tensor): (batch_size, feature_dim) embeddings

    Returns:
        torch.Tensor: Orthogonality loss
    """
    batch_size = features.shape[0]

    # Compute Gram matrix of embeddings
    gram_matrix = torch.mm(features, features.T)

    # Remove diagonal (self-correlations)
    identity = torch.eye(batch_size, device=features.device)
    gram_matrix = gram_matrix - identity

    # Penalize off-diagonal values (correlations)
    loss = (gram_matrix ** 2).mean()
    return loss

def entropy_loss(logits):
    """
    Encourages the classifier to make confident predictions.
    
    Args:
        logits (torch.Tensor): Classifier logits (batch_size, num_classes)
    
    Returns:
        torch.Tensor: Entropy regularization loss
    """
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
    return entropy.mean()

class CombinedLoss(nn.Module):

    def __init__(self, center_loss_fn, alpha=0.5, beta=0.5, lambda_c=0.01, lambda_o=0.01, gamma=0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_c = lambda_c
        self.lambda_o = lambda_o
        self.gamma = gamma
        self.center_loss_fn = center_loss_fn

    def forward(self, features, classification_outputs, labels):
        contrastive_loss_value = contrastive_loss_with_labels(features, labels)

        # Compute classification loss (CrossEntropy)
        logits_cls = classification_outputs
        classification_loss_value = F.cross_entropy(logits_cls, labels)

        # Compute center loss (to improve intra-class compactness)
        center_loss_value = self.center_loss_fn(features, labels)

        # Compute orthogonality loss (to prevent feature collapse)
        ortho_loss_value = orthogonality_loss(features)

        # Compute entropy loss (to ensure confident classification)
        entropy_loss_value = entropy_loss(logits_cls)
        total_loss = (
                self.beta * contrastive_loss_value +  # Contrastive loss
                self.alpha * classification_loss_value +  # Classification loss
                self.lambda_c * center_loss_value +  # Center loss
                self.lambda_o * ortho_loss_value -  # Orthogonality loss
                self.gamma * entropy_loss_value  # Entropy regularization
            )

        # Return total loss + loss breakdown
        loss_dict = {
            "total_loss": total_loss.item(),
            "contrastive_loss": contrastive_loss_value.item(),
            "classification_loss": classification_loss_value.item(),
            "center_loss": center_loss_value.item(),
            "orthogonality_loss": ortho_loss_value.item(),
            "entropy_loss": entropy_loss_value.item(),
        }

        return total_loss

class SculptedContrastiveLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.3, repulsion_weight=1.0):
        """
        Args:
            centers: dict mapping class labels to center vectors (torch tensors).
                     Example: {0: tensor([1, 0, 0]), 1: tensor([-1, 0, 0])}
            margin: minimum distance enforced between embeddings.
            repulsion_weight: weighting factor for the repulsion (contrastive) term.
        """
        super().__init__()
        self.margin = margin
        self.repulsion_weight = repulsion_weight
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        centers = {}
        for class_idx in range(num_classes):
            center = torch.zeros(embedding_dim)
            # Alternate positive and negative directions on each axis for diversity
            direction = 1.0 if class_idx % 2 == 0 else -1.0
            axis = class_idx % embedding_dim  # Wrap around if num_classes > embedding_dim
            center[axis] = direction
            centers[class_idx] = center
        self.register_buffer('centers', torch.stack([centers[i] for i in range(num_classes)]))

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)

        # Attractive term: pull towards class centers
        center_targets = torch.stack([self.centers[int(label.item())] for label in labels]).to(embeddings.device)
        attractive_loss = F.mse_loss(embeddings, center_targets)

        # Repulsive term: enforce margin between embeddings
        repulsive_loss = 0.0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                dist = F.pairwise_distance(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
                # Only enforce margin if labels are different (optional)
                if labels[i] != labels[j]:
                    repulsive_loss += F.relu(self.margin - dist).pow(2)

        # Normalize repulsion by number of pairs
        if batch_size > 1:
            repulsive_loss /= (batch_size * (batch_size - 1) / 2)

        total_loss = attractive_loss + self.repulsion_weight * repulsive_loss
        return total_loss
    
class SupervisedContrastiveEntropyLoss(nn.Module):
    def __init__(
            self, 
            embedding_dim,
            num_classes,
            temperature=0.1, 
            alpha=0.5, 
            beta=0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.sculpted_loss = SculptedContrastiveLoss(
            self.embedding_dim,
            self.num_classes,
            margin=0.5,
            repulsion_weight=1.0
        )

    def forward(self, features, classification, labels):

        classification_loss = F.cross_entropy(classification, labels)
        contrastive_loss_value = self.sculpted_loss(features, labels)

        total_loss = (
                self.beta * contrastive_loss_value +  # Contrastive loss
                self.alpha * classification_loss# Classification loss
            )
        loss_dict = {
            "total_loss": total_loss.item(),
            "contrastive_loss": contrastive_loss_value.item(),
            "classification_loss": contrastive_loss_value.item(),
        }

        return total_loss