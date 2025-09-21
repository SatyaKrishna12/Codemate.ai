"""
Ranking Service for multi-factor scoring and result ranking in the IR system.

This service combines semantic similarity scores with metadata-based signals
such as recency, credibility, user preferences, and document quality to
provide comprehensive ranking for search results.
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import RankedResult

logger = get_logger(__name__)


@dataclass
class RankingFeature:
    """Configuration for a ranking feature."""
    name: str
    weight: float
    enabled: bool = True
    normalization: str = "minmax"  # "minmax", "zscore", "log"


class FeatureExtractor:
    """Extracts and calculates ranking features from search results."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.features = [
            RankingFeature("semantic_similarity", settings.ranking_weights["semantic"], True),
            RankingFeature("recency", settings.ranking_weights["recency"], True),
            RankingFeature("credibility", settings.ranking_weights["credibility"], True),
            RankingFeature("document_quality", settings.ranking_weights["document_quality"], True),
            RankingFeature("user_preferences", settings.ranking_weights["user_preferences"], True),
        ]
    
    def extract_semantic_score(self, result: RankedResult) -> float:
        """Extract semantic similarity score."""
        return result.semantic_score
    
    def extract_recency_score(self, result: RankedResult) -> float:
        """Calculate recency score based on document creation date."""
        try:
            created_at = result.source_metadata.get('created_at')
            if not created_at:
                return 0.5  # Neutral score for unknown dates
            
            if isinstance(created_at, str):
                # Parse date string
                try:
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    # Try other common formats
                    try:
                        created_date = datetime.strptime(created_at, '%Y-%m-%d')
                    except:
                        return 0.5
            elif isinstance(created_at, datetime):
                created_date = created_at
            else:
                return 0.5
            
            # Calculate age in days
            age_days = (datetime.now() - created_date.replace(tzinfo=None)).days
            
            # Exponential decay: more recent = higher score
            # Score decreases by half every 365 days (1 year)
            half_life_days = 365
            decay_factor = math.exp(-math.log(2) * age_days / half_life_days)
            
            return min(decay_factor, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating recency score: {e}")
            return 0.5
    
    def extract_credibility_score(self, result: RankedResult) -> float:
        """Calculate credibility score based on source and document metadata."""
        try:
            metadata = result.source_metadata
            score = 0.5  # Base score
            
            # Source type credibility
            source_type = metadata.get('source_type', '').lower()
            if source_type in ['official', 'government', 'academic']:
                score += 0.3
            elif source_type in ['news', 'journal', 'documentation']:
                score += 0.2
            elif source_type in ['blog', 'forum', 'social']:
                score += 0.1
            
            # Domain credibility (if available)
            domain = metadata.get('domain', '')
            trusted_domains = [
                'edu', 'gov', 'org', 'arxiv.org', 'ieee.org', 
                'acm.org', 'nih.gov', 'github.com'
            ]
            if any(trusted in domain for trusted in trusted_domains):
                score += 0.2
            
            # Author credibility (if available)
            author = metadata.get('author', '')
            if author and len(author) > 0:
                score += 0.1  # Bonus for having author information
            
            # Citation count (if available)
            citations = metadata.get('citation_count', 0)
            if citations > 0:
                # Log scale for citations
                citation_score = min(math.log10(citations + 1) / 3, 0.2)
                score += citation_score
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating credibility score: {e}")
            return 0.5
    
    def extract_document_quality_score(self, result: RankedResult) -> float:
        """Calculate document quality score based on content and metadata."""
        try:
            metadata = result.source_metadata
            content = result.content_snippet
            score = 0.5  # Base score
            
            # Content length quality (not too short, not too long)
            content_length = len(content)
            if 100 <= content_length <= 2000:
                score += 0.2
            elif 50 <= content_length < 100 or 2000 < content_length <= 5000:
                score += 0.1
            
            # Structure quality indicators
            if any(indicator in content.lower() for indicator in ['introduction', 'conclusion', 'abstract']):
                score += 0.1
            
            # Has proper formatting (bullet points, numbers, etc.)
            if any(char in content for char in ['•', '-', '1.', '2.', '*']):
                score += 0.05
            
            # Language quality (simple heuristics)
            words = content.split()
            if len(words) > 10:
                avg_word_length = sum(len(word) for word in words) / len(words)
                if 4 <= avg_word_length <= 8:  # Good vocabulary range
                    score += 0.1
            
            # Document type quality
            doc_type = metadata.get('document_type', '').lower()
            if doc_type in ['research_paper', 'documentation', 'manual']:
                score += 0.15
            elif doc_type in ['article', 'report', 'guide']:
                score += 0.1
            
            # Has metadata completeness
            metadata_fields = ['title', 'author', 'created_at', 'source']
            present_fields = sum(1 for field in metadata_fields if metadata.get(field))
            metadata_completeness = present_fields / len(metadata_fields)
            score += metadata_completeness * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating document quality score: {e}")
            return 0.5
    
    def extract_user_preferences_score(
        self, 
        result: RankedResult, 
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate user preference score based on user history and preferences."""
        try:
            if not user_preferences:
                return 0.5  # Neutral score when no preferences available
            
            metadata = result.source_metadata
            score = 0.5  # Base score
            
            # Preferred sources
            preferred_sources = user_preferences.get('preferred_sources', [])
            source = metadata.get('source', '')
            if any(pref_source in source for pref_source in preferred_sources):
                score += 0.2
            
            # Preferred document types
            preferred_types = user_preferences.get('preferred_document_types', [])
            doc_type = metadata.get('document_type', '')
            if doc_type in preferred_types:
                score += 0.15
            
            # Language preferences
            preferred_language = user_preferences.get('preferred_language', 'english')
            content_language = metadata.get('language', 'english')
            if content_language == preferred_language:
                score += 0.1
            
            # Recency preference
            recency_preference = user_preferences.get('recency_preference', 'balanced')
            if recency_preference == 'recent':
                # Boost recent documents more
                recency_score = self.extract_recency_score(result)
                score += recency_score * 0.2
            elif recency_preference == 'classic':
                # Boost older, established documents
                recency_score = self.extract_recency_score(result)
                score += (1 - recency_score) * 0.15
            
            # Topic interests (if available)
            user_topics = user_preferences.get('interested_topics', [])
            content = result.content_snippet.lower()
            topic_matches = sum(1 for topic in user_topics if topic.lower() in content)
            if topic_matches > 0:
                topic_boost = min(topic_matches * 0.05, 0.15)
                score += topic_boost
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating user preferences score: {e}")
            return 0.5


class ScoreNormalizer:
    """Normalizes feature scores across different scales."""
    
    @staticmethod
    def minmax_normalize(scores: List[float]) -> List[float]:
        """Normalize scores using min-max normalization."""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if min_score == max_score:
            return [0.5] * len(scores)  # All equal scores
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    @staticmethod
    def zscore_normalize(scores: List[float]) -> List[float]:
        """Normalize scores using z-score normalization."""
        if not scores:
            return scores
        
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return [0.5] * len(scores)  # All equal scores
        
        z_scores = [(score - mean_score) / std_dev for score in scores]
        # Convert z-scores to 0-1 range using sigmoid
        return [1 / (1 + math.exp(-z)) for z in z_scores]
    
    @staticmethod
    def log_normalize(scores: List[float]) -> List[float]:
        """Normalize scores using logarithmic scaling."""
        if not scores:
            return scores
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        log_scores = [math.log(score + epsilon) for score in scores]
        return ScoreNormalizer.minmax_normalize(log_scores)


class RankingService:
    """
    Multi-factor ranking service that combines semantic similarity with
    metadata-based signals for comprehensive result ranking.
    """
    
    def __init__(self):
        """Initialize the ranking service."""
        self.feature_extractor = FeatureExtractor()
        self.normalizer = ScoreNormalizer()
        
    def rank_results(
        self, 
        results: List[RankedResult],
        user_preferences: Optional[Dict[str, Any]] = None,
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[RankedResult]:
        """
        Rank search results using multi-factor scoring.
        
        Args:
            results: List of search results to rank
            user_preferences: Optional user preferences for personalization
            query_context: Optional query context for query-specific ranking
            
        Returns:
            Ranked list of results with updated scores
        """
        if not results:
            return results
        
        try:
            logger.info(f"Ranking {len(results)} search results")
            
            # Extract all features for all results
            feature_scores = self._extract_all_features(results, user_preferences)
            
            # Normalize features
            normalized_features = self._normalize_features(feature_scores)
            
            # Calculate final scores
            ranked_results = self._calculate_final_scores(results, normalized_features)
            
            # Sort by final score
            ranked_results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Update rankings and explanations
            for i, result in enumerate(ranked_results):
                result.relevance_explanation['final_rank'] = i + 1
                result.relevance_explanation['ranking_features'] = normalized_features[i]
            
            logger.info(f"Ranking completed. Top score: {ranked_results[0].final_score:.3f}")
            return ranked_results
            
        except Exception as e:
            logger.error(f"Error in ranking results: {e}")
            # Return original results if ranking fails
            return results
    
    def _extract_all_features(
        self, 
        results: List[RankedResult],
        user_preferences: Optional[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Extract all ranking features for all results."""
        feature_scores = defaultdict(list)
        
        for result in results:
            feature_scores['semantic'].append(
                self.feature_extractor.extract_semantic_score(result)
            )
            feature_scores['recency'].append(
                self.feature_extractor.extract_recency_score(result)
            )
            feature_scores['credibility'].append(
                self.feature_extractor.extract_credibility_score(result)
            )
            feature_scores['quality'].append(
                self.feature_extractor.extract_document_quality_score(result)
            )
            feature_scores['preferences'].append(
                self.feature_extractor.extract_user_preferences_score(result, user_preferences)
            )
        
        return feature_scores
    
    def _normalize_features(self, feature_scores: Dict[str, List[float]]) -> List[Dict[str, float]]:
        """Normalize feature scores and organize by result."""
        normalized_by_feature = {}
        
        # Normalize each feature separately
        for feature_name, scores in feature_scores.items():
            # Get normalization method for this feature
            feature_config = next(
                (f for f in self.feature_extractor.features if feature_name in f.name), 
                None
            )
            
            if feature_config and feature_config.enabled:
                if feature_config.normalization == "minmax":
                    normalized_scores = self.normalizer.minmax_normalize(scores)
                elif feature_config.normalization == "zscore":
                    normalized_scores = self.normalizer.zscore_normalize(scores)
                elif feature_config.normalization == "log":
                    normalized_scores = self.normalizer.log_normalize(scores)
                else:
                    normalized_scores = scores  # No normalization
            else:
                normalized_scores = [0.0] * len(scores)  # Disabled feature
            
            normalized_by_feature[feature_name] = normalized_scores
        
        # Reorganize by result index
        num_results = len(next(iter(feature_scores.values())))
        normalized_by_result = []
        
        for i in range(num_results):
            result_features = {}
            for feature_name, normalized_scores in normalized_by_feature.items():
                result_features[feature_name] = normalized_scores[i]
            normalized_by_result.append(result_features)
        
        return normalized_by_result
    
    def _calculate_final_scores(
        self, 
        results: List[RankedResult], 
        normalized_features: List[Dict[str, float]]
    ) -> List[RankedResult]:
        """Calculate final weighted scores for each result."""
        updated_results = []
        
        for i, result in enumerate(results):
            features = normalized_features[i]
            
            # Calculate weighted final score
            final_score = 0.0
            score_breakdown = {}
            
            for feature_config in self.feature_extractor.features:
                if feature_config.enabled and any(name in feature_config.name for name in features.keys()):
                    # Find matching feature score
                    feature_score = 0.0
                    for feature_name, score in features.items():
                        if feature_name in feature_config.name:
                            feature_score = score
                            break
                    
                    weighted_score = feature_score * feature_config.weight
                    final_score += weighted_score
                    score_breakdown[feature_config.name] = {
                        'raw_score': feature_score,
                        'weight': feature_config.weight,
                        'weighted_score': weighted_score
                    }
            
            # Update result with new scores
            result.final_score = final_score
            result.metadata_score = final_score - result.semantic_score  # Metadata contribution
            result.relevance_explanation['score_breakdown'] = score_breakdown
            result.relevance_explanation['total_weighted_score'] = final_score
            
            updated_results.append(result)
        
        return updated_results
    
    def get_ranking_stats(self) -> Dict[str, Any]:
        """Get ranking service statistics."""
        enabled_features = [f.name for f in self.feature_extractor.features if f.enabled]
        feature_weights = {f.name: f.weight for f in self.feature_extractor.features}
        
        return {
            "enabled_features": enabled_features,
            "feature_weights": feature_weights,
            "total_weight": sum(f.weight for f in self.feature_extractor.features if f.enabled),
            "normalization_methods": {f.name: f.normalization for f in self.feature_extractor.features}
        }


# Global instance
ranking_service = RankingService()
