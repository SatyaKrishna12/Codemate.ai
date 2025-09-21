"""
Aggregation Service for merging, deduplicating, and normalizing results from multiple retrieval strategies.

This service combines results from different retrieval methods, removes duplicates,
handles result fusion, and provides unified result sets with proper attribution
to the original retrieval strategies.
"""

import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import difflib

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import RankedResult

logger = get_logger(__name__)


@dataclass
class AggregationStrategy:
    """Configuration for result aggregation strategy."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    parameters: Dict[str, Any] = None


class ContentSimilarityCalculator:
    """Calculates content similarity for deduplication."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize with similarity threshold."""
        self.similarity_threshold = similarity_threshold
    
    def calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        try:
            # Use difflib for sequence matching
            similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
            return similarity
        except Exception as e:
            logger.warning(f"Error calculating content similarity: {e}")
            return 0.0
    
    def are_similar(self, content1: str, content2: str) -> bool:
        """Check if two content pieces are similar enough to be considered duplicates."""
        similarity = self.calculate_similarity(content1, content2)
        return similarity >= self.similarity_threshold
    
    def get_content_hash(self, content: str, length: int = 100) -> str:
        """Generate a hash for content comparison."""
        # Normalize content: remove extra whitespace, convert to lowercase
        normalized = ' '.join(content.lower().strip().split())
        # Use first 'length' characters for hash
        sample = normalized[:length]
        return hashlib.md5(sample.encode()).hexdigest()


class DuplicationDetector:
    """Detects and manages duplicate results."""
    
    def __init__(self, similarity_threshold: float = None):
        """Initialize duplication detector."""
        self.similarity_threshold = similarity_threshold or settings.deduplication_threshold
        self.similarity_calculator = ContentSimilarityCalculator(self.similarity_threshold)
    
    def find_duplicates(self, results: List[RankedResult]) -> Dict[str, List[int]]:
        """
        Find duplicate results and group them.
        
        Args:
            results: List of results to check for duplicates
            
        Returns:
            Dictionary mapping group_id to list of result indices
        """
        duplicate_groups = {}
        processed_indices = set()
        group_counter = 0
        
        for i, result1 in enumerate(results):
            if i in processed_indices:
                continue
            
            # Start a new group with this result
            current_group = [i]
            processed_indices.add(i)
            
            # Find all similar results
            for j, result2 in enumerate(results[i+1:], start=i+1):
                if j in processed_indices:
                    continue
                
                if self._are_duplicates(result1, result2):
                    current_group.append(j)
                    processed_indices.add(j)
            
            # Only create group if there are duplicates
            if len(current_group) > 1:
                group_id = f"group_{group_counter}"
                duplicate_groups[group_id] = current_group
                group_counter += 1
        
        return duplicate_groups
    
    def _are_duplicates(self, result1: RankedResult, result2: RankedResult) -> bool:
        """Check if two results are duplicates."""
        # Same chunk ID - definitely duplicates
        if result1.chunk_id == result2.chunk_id:
            return True
        
        # Same document ID and very similar content
        if (result1.document_id == result2.document_id and 
            self.similarity_calculator.are_similar(result1.content_snippet, result2.content_snippet)):
            return True
        
        # Very similar content from any source
        if self.similarity_calculator.are_similar(result1.content_snippet, result2.content_snippet):
            # Additional check: similar position or metadata
            pos1 = result1.position_in_document or 0
            pos2 = result2.position_in_document or 0
            if abs(pos1 - pos2) < 5:  # Close positions
                return True
        
        return False


class ResultFuser:
    """Fuses multiple results using different strategies."""
    
    def __init__(self):
        """Initialize result fuser."""
        self.fusion_strategies = {
            'reciprocal_rank': self._reciprocal_rank_fusion,
            'weighted_average': self._weighted_average_fusion,
            'max_score': self._max_score_fusion,
            'borda_count': self._borda_count_fusion
        }
    
    def fuse_results(
        self, 
        result_lists: List[Tuple[str, List[RankedResult]]], 
        strategy: str = 'reciprocal_rank'
    ) -> List[RankedResult]:
        """
        Fuse multiple result lists using specified strategy.
        
        Args:
            result_lists: List of (strategy_name, results) tuples
            strategy: Fusion strategy to use
            
        Returns:
            Fused and ranked result list
        """
        if not result_lists:
            return []
        
        if strategy not in self.fusion_strategies:
            logger.warning(f"Unknown fusion strategy: {strategy}. Using reciprocal_rank")
            strategy = 'reciprocal_rank'
        
        try:
            return self.fusion_strategies[strategy](result_lists)
        except Exception as e:
            logger.error(f"Error in result fusion: {e}")
            # Fallback: return first result list
            return result_lists[0][1] if result_lists else []
    
    def _reciprocal_rank_fusion(
        self, 
        result_lists: List[Tuple[str, List[RankedResult]]]
    ) -> List[RankedResult]:
        """Fuse results using reciprocal rank fusion (RRF)."""
        # Collect all unique results
        all_results = {}  # chunk_id -> RankedResult
        rrf_scores = defaultdict(float)
        k = 60  # RRF parameter
        
        for strategy_name, results in result_lists:
            for rank, result in enumerate(results, 1):
                result_id = result.chunk_id
                
                # Store result if not seen before
                if result_id not in all_results:
                    all_results[result_id] = result
                
                # Add RRF score: 1/(k + rank)
                rrf_score = 1 / (k + rank)
                rrf_scores[result_id] += rrf_score
                
                # Update result metadata
                if 'fusion_info' not in result.relevance_explanation:
                    result.relevance_explanation['fusion_info'] = {}
                result.relevance_explanation['fusion_info'][strategy_name] = {
                    'rank': rank,
                    'rrf_contribution': rrf_score
                }
        
        # Update final scores and sort
        fused_results = []
        for result_id, result in all_results.items():
            result.final_score = rrf_scores[result_id]
            result.relevance_explanation['fusion_strategy'] = 'reciprocal_rank'
            result.relevance_explanation['rrf_score'] = rrf_scores[result_id]
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results
    
    def _weighted_average_fusion(
        self, 
        result_lists: List[Tuple[str, List[RankedResult]]]
    ) -> List[RankedResult]:
        """Fuse results using weighted average of scores."""
        all_results = {}
        weighted_scores = defaultdict(float)
        weight_sums = defaultdict(float)
        
        # Get strategy weights from settings
        strategy_weights = getattr(settings, 'ensemble_retriever_weights', [])
        if len(strategy_weights) != len(result_lists):
            # Use equal weights if not configured properly
            strategy_weights = [1.0 / len(result_lists)] * len(result_lists)
        
        for i, (strategy_name, results) in enumerate(result_lists):
            weight = strategy_weights[i] if i < len(strategy_weights) else 1.0
            
            for result in results:
                result_id = result.chunk_id
                
                if result_id not in all_results:
                    all_results[result_id] = result
                
                weighted_scores[result_id] += result.final_score * weight
                weight_sums[result_id] += weight
                
                # Update metadata
                if 'fusion_info' not in result.relevance_explanation:
                    result.relevance_explanation['fusion_info'] = {}
                result.relevance_explanation['fusion_info'][strategy_name] = {
                    'score': result.final_score,
                    'weight': weight
                }
        
        # Calculate final weighted averages
        fused_results = []
        for result_id, result in all_results.items():
            if weight_sums[result_id] > 0:
                result.final_score = weighted_scores[result_id] / weight_sums[result_id]
            result.relevance_explanation['fusion_strategy'] = 'weighted_average'
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results
    
    def _max_score_fusion(
        self, 
        result_lists: List[Tuple[str, List[RankedResult]]]
    ) -> List[RankedResult]:
        """Fuse results by taking maximum score for each result."""
        all_results = {}
        max_scores = defaultdict(float)
        
        for strategy_name, results in result_lists:
            for result in results:
                result_id = result.chunk_id
                
                if result_id not in all_results:
                    all_results[result_id] = result
                
                if result.final_score > max_scores[result_id]:
                    max_scores[result_id] = result.final_score
                    # Update the stored result with the best scoring version
                    all_results[result_id] = result
        
        # Update final scores
        fused_results = []
        for result_id, result in all_results.items():
            result.final_score = max_scores[result_id]
            result.relevance_explanation['fusion_strategy'] = 'max_score'
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results
    
    def _borda_count_fusion(
        self, 
        result_lists: List[Tuple[str, List[RankedResult]]]
    ) -> List[RankedResult]:
        """Fuse results using Borda count voting."""
        all_results = {}
        borda_scores = defaultdict(int)
        
        for strategy_name, results in result_lists:
            n = len(results)
            for rank, result in enumerate(results):
                result_id = result.chunk_id
                
                if result_id not in all_results:
                    all_results[result_id] = result
                
                # Borda score: n - rank (higher is better)
                borda_score = n - rank
                borda_scores[result_id] += borda_score
        
        # Update final scores (normalize by max possible score)
        max_possible_score = sum(len(results) for _, results in result_lists)
        fused_results = []
        
        for result_id, result in all_results.items():
            normalized_score = borda_scores[result_id] / max_possible_score if max_possible_score > 0 else 0
            result.final_score = normalized_score
            result.relevance_explanation['fusion_strategy'] = 'borda_count'
            result.relevance_explanation['borda_score'] = borda_scores[result_id]
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results


class AggregationService:
    """
    Service for aggregating results from multiple retrieval strategies with
    deduplication, fusion, and normalization capabilities.
    """
    
    def __init__(self):
        """Initialize the aggregation service."""
        self.duplication_detector = DuplicationDetector()
        self.result_fuser = ResultFuser()
        self.strategies = [
            AggregationStrategy("deduplication", True, 1.0),
            AggregationStrategy("fusion", True, 1.0, {"strategy": "reciprocal_rank"}),
            AggregationStrategy("normalization", True, 1.0)
        ]
    
    def aggregate_results(
        self,
        result_lists: List[Tuple[str, List[RankedResult]]],
        max_results: int = 50,
        fusion_strategy: str = 'reciprocal_rank',
        deduplicate: bool = True
    ) -> List[RankedResult]:
        """
        Aggregate multiple result lists into a unified, deduplicated, and ranked list.
        
        Args:
            result_lists: List of (strategy_name, results) tuples
            max_results: Maximum number of results to return
            fusion_strategy: Strategy for fusing multiple result lists
            deduplicate: Whether to remove duplicate results
            
        Returns:
            Aggregated and ranked result list
        """
        if not result_lists:
            return []
        
        try:
            logger.info(f"Aggregating {len(result_lists)} result lists")
            
            # Step 1: Fuse results from multiple strategies
            fused_results = self.result_fuser.fuse_results(result_lists, fusion_strategy)
            logger.info(f"Fusion completed: {len(fused_results)} results")
            
            # Step 2: Deduplicate if requested
            if deduplicate:
                deduplicated_results = self._deduplicate_results(fused_results)
                logger.info(f"Deduplication completed: {len(deduplicated_results)} results")
            else:
                deduplicated_results = fused_results
            
            # Step 3: Normalize scores
            normalized_results = self._normalize_scores(deduplicated_results)
            
            # Step 4: Limit results
            final_results = normalized_results[:max_results]
            
            # Step 5: Add aggregation metadata
            for i, result in enumerate(final_results):
                result.relevance_explanation['aggregation_rank'] = i + 1
                result.relevance_explanation['original_lists_count'] = len(result_lists)
                result.relevance_explanation['post_aggregation_score'] = result.final_score
            
            logger.info(f"Aggregation completed: {len(final_results)} final results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in result aggregation: {e}")
            # Fallback: return first result list with basic deduplication
            if result_lists:
                fallback_results = result_lists[0][1]
                return self._deduplicate_results(fallback_results)[:max_results]
            return []
    
    def _deduplicate_results(self, results: List[RankedResult]) -> List[RankedResult]:
        """Remove duplicate results from the list."""
        if not results:
            return results
        
        # Find duplicate groups
        duplicate_groups = self.duplication_detector.find_duplicates(results)
        
        # Keep track of indices to remove
        indices_to_remove = set()
        
        for group_id, indices in duplicate_groups.items():
            if len(indices) <= 1:
                continue
            
            # Find the best result in the group (highest score)
            best_idx = max(indices, key=lambda i: results[i].final_score)
            
            # Mark others for removal
            for idx in indices:
                if idx != best_idx:
                    indices_to_remove.add(idx)
            
            # Update the best result with deduplication info
            best_result = results[best_idx]
            if 'deduplication_info' not in best_result.relevance_explanation:
                best_result.relevance_explanation['deduplication_info'] = {}
            
            best_result.relevance_explanation['deduplication_info'] = {
                'group_id': group_id,
                'duplicates_removed': len(indices) - 1,
                'duplicate_sources': [
                    results[idx].source_metadata.get('source', 'unknown') 
                    for idx in indices if idx != best_idx
                ]
            }
        
        # Create deduplicated list
        deduplicated = [
            result for i, result in enumerate(results) 
            if i not in indices_to_remove
        ]
        
        logger.info(f"Removed {len(indices_to_remove)} duplicate results")
        return deduplicated
    
    def _normalize_scores(self, results: List[RankedResult]) -> List[RankedResult]:
        """Normalize final scores to 0-1 range."""
        if not results:
            return results
        
        scores = [result.final_score for result in results]
        
        if not scores:
            return results
        
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if min_score == max_score:
            for result in results:
                result.final_score = 0.5
        else:
            for result in results:
                normalized_score = (result.final_score - min_score) / (max_score - min_score)
                result.relevance_explanation['pre_normalization_score'] = result.final_score
                result.final_score = normalized_score
        
        return results
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation service statistics."""
        return {
            "available_fusion_strategies": list(self.result_fuser.fusion_strategies.keys()),
            "deduplication_threshold": self.duplication_detector.similarity_threshold,
            "enabled_strategies": [s.name for s in self.strategies if s.enabled],
            "default_fusion_strategy": "reciprocal_rank"
        }


# Global instance
aggregation_service = AggregationService()
