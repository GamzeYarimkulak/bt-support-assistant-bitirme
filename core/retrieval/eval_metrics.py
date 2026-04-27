"""
Evaluation metrics for retrieval quality assessment.
"""

from typing import List, Set, Dict, Any
import numpy as np
import structlog

logger = structlog.get_logger()


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Precision@k.
    
    Args:
        retrieved: List of retrieved document IDs (ordered)
        relevant: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        Precision@k score
    """
    if k <= 0 or not retrieved:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    
    return relevant_retrieved / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Recall@k.
    
    Args:
        retrieved: List of retrieved document IDs (ordered)
        relevant: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        Recall@k score
    """
    if not relevant or k <= 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    
    return relevant_retrieved / len(relevant)


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Average Precision (AP).
    
    Args:
        retrieved: List of retrieved document IDs (ordered)
        relevant: Set of relevant document IDs
        
    Returns:
        Average precision score
    """
    if not relevant or not retrieved:
        return 0.0
    
    relevant_count = 0
    precision_sum = 0.0
    
    for k, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            relevant_count += 1
            precision_sum += relevant_count / k
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / len(relevant)


def mean_average_precision(
    retrieved_lists: List[List[str]], 
    relevant_sets: List[Set[str]]
) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.
    
    Args:
        retrieved_lists: List of retrieved document ID lists (one per query)
        relevant_sets: List of relevant document ID sets (one per query)
        
    Returns:
        MAP score
    """
    if not retrieved_lists or len(retrieved_lists) != len(relevant_sets):
        return 0.0
    
    aps = [
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_lists, relevant_sets)
    ]
    
    return np.mean(aps)


def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.
    
    Args:
        relevances: List of relevance scores (ordered by retrieval rank)
        k: Cutoff position
        
    Returns:
        DCG@k score
    """
    if k <= 0 or not relevances:
        return 0.0
    
    relevances_at_k = relevances[:k]
    
    dcg = relevances_at_k[0]
    for i, rel in enumerate(relevances_at_k[1:], 2):
        dcg += rel / np.log2(i)
    
    return dcg


def ndcg_at_k(relevances: List[float], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        relevances: List of relevance scores (ordered by retrieval rank)
        k: Cutoff position
        
    Returns:
        nDCG@k score
    """
    if k <= 0 or not relevances:
        return 0.0
    
    dcg = dcg_at_k(relevances, k)
    
    # Ideal DCG (sort relevances in descending order)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def mean_reciprocal_rank(retrieved_lists: List[List[str]], relevant_sets: List[Set[str]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple queries.
    
    Args:
        retrieved_lists: List of retrieved document ID lists (one per query)
        relevant_sets: List of relevant document ID sets (one per query)
        
    Returns:
        MRR score
    """
    if not retrieved_lists or len(retrieved_lists) != len(relevant_sets):
        return 0.0
    
    reciprocal_ranks = []
    
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    
    return np.mean(reciprocal_ranks)


def evaluate_retrieval(
    retrieved: List[str],
    relevant: Set[str],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a single retrieval result.
    
    Args:
        retrieved: List of retrieved document IDs (ordered)
        relevant: Set of relevant document IDs
        k_values: List of k values for P@k, R@k metrics
        
    Returns:
        Dictionary with all evaluation metrics
    """
    metrics = {
        "average_precision": average_precision(retrieved, relevant)
    }
    
    for k in k_values:
        metrics[f"precision_at_{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"recall_at_{k}"] = recall_at_k(retrieved, relevant, k)
    
    # Reciprocal rank
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            metrics["reciprocal_rank"] = 1.0 / rank
            break
    else:
        metrics["reciprocal_rank"] = 0.0
    
    return metrics


