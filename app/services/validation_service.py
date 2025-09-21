"""
Validation Service for cross-referencing information across sources to detect consistency/conflicts.

This service analyzes search results to identify supporting evidence, contradictions,
and builds confidence scores based on source agreement, credibility, and consistency
across multiple documents and sources.
"""

import re
import asyncio
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import RankedResult, ValidationResult, ValidationReport
from app.services.embedding_service import embedding_service

logger = get_logger(__name__)


@dataclass
class ClaimExtraction:
    """Represents an extracted claim from content."""
    text: str
    confidence: float
    source_chunk_id: str
    extraction_method: str
    keywords: List[str]


@dataclass
class ConsistencyCheck:
    """Results of consistency checking between sources."""
    claim1: ClaimExtraction
    claim2: ClaimExtraction
    similarity_score: float
    consistency_type: str  # "supporting", "contradicting", "neutral"
    confidence: float


class ClaimExtractor:
    """Extracts factual claims from text content."""
    
    def __init__(self):
        """Initialize claim extractor."""
        # Patterns for identifying factual statements
        self.fact_patterns = [
            r'(\w+\s+is\s+\w+)',  # X is Y
            r'(\w+\s+was\s+\w+)',  # X was Y
            r'(\w+\s+has\s+\w+)',  # X has Y
            r'(\w+\s+contains?\s+\w+)',  # X contains Y
            r'(\w+\s+shows?\s+\w+)',  # X shows Y
            r'(\w+\s+indicates?\s+\w+)',  # X indicates Y
            r'(\w+\s+demonstrates?\s+\w+)',  # X demonstrates Y
            r'(According to \w+)',  # According to X
            r'(Research shows that)',  # Research shows that
            r'(Studies indicate that)',  # Studies indicate that
        ]
        
        # Confidence indicators
        self.high_confidence_words = [
            'proven', 'demonstrated', 'confirmed', 'established', 'verified',
            'documented', 'research shows', 'studies indicate', 'data shows'
        ]
        
        self.low_confidence_words = [
            'suggests', 'implies', 'may be', 'could be', 'appears to',
            'seems to', 'possibly', 'probably', 'likely'
        ]
    
    def extract_claims(self, content: str, chunk_id: str) -> List[ClaimExtraction]:
        """
        Extract factual claims from content.
        
        Args:
            content: Text content to analyze
            chunk_id: Identifier for the content chunk
            
        Returns:
            List of extracted claims
        """
        claims = []
        
        try:
            # Split content into sentences
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                
                # Check if sentence contains factual patterns
                for pattern in self.fact_patterns:
                    matches = re.findall(pattern, sentence, re.IGNORECASE)
                    if matches:
                        claim_text = sentence
                        confidence = self._calculate_claim_confidence(sentence)
                        keywords = self._extract_keywords(sentence)
                        
                        claim = ClaimExtraction(
                            text=claim_text,
                            confidence=confidence,
                            source_chunk_id=chunk_id,
                            extraction_method="pattern_matching",
                            keywords=keywords
                        )
                        claims.append(claim)
                        break  # Don't double-count same sentence
            
            # Additional extraction: Look for numerical facts
            numerical_claims = self._extract_numerical_facts(content, chunk_id)
            claims.extend(numerical_claims)
            
            return claims[:10]  # Limit to top 10 claims per chunk
            
        except Exception as e:
            logger.warning(f"Error extracting claims from {chunk_id}: {e}")
            return []
    
    def _calculate_claim_confidence(self, sentence: str) -> float:
        """Calculate confidence score for a claim based on language indicators."""
        sentence_lower = sentence.lower()
        confidence = 0.5  # Base confidence
        
        # Check for high confidence indicators
        for indicator in self.high_confidence_words:
            if indicator in sentence_lower:
                confidence += 0.1
        
        # Check for low confidence indicators
        for indicator in self.low_confidence_words:
            if indicator in sentence_lower:
                confidence -= 0.1
        
        # Longer sentences with specific details tend to be more factual
        if len(sentence) > 100:
            confidence += 0.05
        
        # Sentences with numbers/data
        if re.search(r'\d+', sentence):
            confidence += 0.1
        
        return max(0.1, min(0.9, confidence))
    
    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract key terms from a sentence."""
        # Simple keyword extraction - could be enhanced with NLP
        words = re.findall(r'\b[A-Za-z]{4,}\b', sentence)
        # Remove common words
        stop_words = {'this', 'that', 'with', 'have', 'will', 'been', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}
        keywords = [word.lower() for word in words if word.lower() not in stop_words]
        return list(set(keywords))[:5]  # Top 5 unique keywords
    
    def _extract_numerical_facts(self, content: str, chunk_id: str) -> List[ClaimExtraction]:
        """Extract numerical/statistical facts from content."""
        numerical_claims = []
        
        # Patterns for numerical facts
        numerical_patterns = [
            r'(\d+(?:\.\d+)?%\s+of\s+\w+)',  # X% of Y
            r'(\$\d+(?:,\d{3})*(?:\.\d{2})?)',  # Dollar amounts
            r'(\d+(?:,\d{3})*\s+(?:people|users|cases|instances))',  # Counts
            r'(\d+(?:\.\d+)?\s+(?:million|billion|thousand))',  # Large numbers
        ]
        
        for pattern in numerical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Find the sentence containing this fact
                sentences = re.split(r'[.!?]+', content)
                for sentence in sentences:
                    if match in sentence:
                        claim = ClaimExtraction(
                            text=sentence.strip(),
                            confidence=0.7,  # Numerical facts tend to be more reliable
                            source_chunk_id=chunk_id,
                            extraction_method="numerical_extraction",
                            keywords=self._extract_keywords(sentence)
                        )
                        numerical_claims.append(claim)
                        break
        
        return numerical_claims


class ConsistencyAnalyzer:
    """Analyzes consistency between claims from different sources."""
    
    def __init__(self):
        """Initialize consistency analyzer."""
        self.similarity_threshold = 0.7
        self.contradiction_keywords = [
            'however', 'but', 'although', 'despite', 'contrary', 'opposite',
            'not', 'never', 'incorrect', 'wrong', 'false', 'disputed'
        ]
    
    async def analyze_consistency(
        self, 
        claims: List[ClaimExtraction]
    ) -> List[ConsistencyCheck]:
        """
        Analyze consistency between claims from different sources.
        
        Args:
            claims: List of extracted claims to analyze
            
        Returns:
            List of consistency check results
        """
        consistency_results = []
        
        try:
            # Group claims by topic/keywords for more efficient comparison
            topic_groups = self._group_claims_by_topic(claims)
            
            for topic, topic_claims in topic_groups.items():
                if len(topic_claims) < 2:
                    continue  # Need at least 2 claims to compare
                
                # Compare claims within each topic group
                for i, claim1 in enumerate(topic_claims):
                    for claim2 in topic_claims[i+1:]:
                        # Skip claims from same source
                        if claim1.source_chunk_id == claim2.source_chunk_id:
                            continue
                        
                        consistency = await self._compare_claims(claim1, claim2)
                        if consistency:
                            consistency_results.append(consistency)
            
            return consistency_results
            
        except Exception as e:
            logger.error(f"Error in consistency analysis: {e}")
            return []
    
    def _group_claims_by_topic(self, claims: List[ClaimExtraction]) -> Dict[str, List[ClaimExtraction]]:
        """Group claims by shared keywords/topics."""
        topic_groups = defaultdict(list)
        
        for claim in claims:
            # Use top keywords as topic identifier
            if claim.keywords:
                topic_key = "_".join(sorted(claim.keywords[:2]))  # Use top 2 keywords
                topic_groups[topic_key].append(claim)
        
        # Remove groups with only one claim
        return {k: v for k, v in topic_groups.items() if len(v) > 1}
    
    async def _compare_claims(
        self, 
        claim1: ClaimExtraction, 
        claim2: ClaimExtraction
    ) -> Optional[ConsistencyCheck]:
        """Compare two claims for consistency."""
        try:
            # Calculate semantic similarity using embeddings
            similarity_score = await self._calculate_semantic_similarity(
                claim1.text, claim2.text
            )
            
            if similarity_score < 0.3:
                return None  # Too dissimilar to be relevant
            
            # Determine consistency type
            consistency_type = self._determine_consistency_type(
                claim1.text, claim2.text, similarity_score
            )
            
            # Calculate overall confidence
            confidence = (claim1.confidence + claim2.confidence) / 2
            confidence *= similarity_score  # Weight by similarity
            
            return ConsistencyCheck(
                claim1=claim1,
                claim2=claim2,
                similarity_score=similarity_score,
                consistency_type=consistency_type,
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"Error comparing claims: {e}")
            return None
    
    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            # Generate embeddings for both texts
            embedding1 = await embedding_service.embed_query(text1)
            embedding2 = await embedding_service.embed_query(text2)
            
            # Calculate cosine similarity
            import numpy as np
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _determine_consistency_type(
        self, 
        text1: str, 
        text2: str, 
        similarity_score: float
    ) -> str:
        """Determine if claims are supporting, contradicting, or neutral."""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Look for contradiction indicators
        contradiction_count = sum(
            1 for keyword in self.contradiction_keywords 
            if keyword in text1_lower or keyword in text2_lower
        )
        
        # High similarity with contradiction keywords = likely contradiction
        if contradiction_count > 0 and similarity_score > 0.6:
            return "contradicting"
        
        # High similarity without contradiction = likely supporting
        elif similarity_score > 0.7:
            return "supporting"
        
        # Medium similarity = neutral/related
        else:
            return "neutral"


class ValidationService:
    """
    Service for validating information consistency across multiple sources
    and building confidence scores based on cross-referencing.
    """
    
    def __init__(self):
        """Initialize validation service."""
        self.claim_extractor = ClaimExtractor()
        self.consistency_analyzer = ConsistencyAnalyzer()
        self.enabled = getattr(settings, 'cross_reference_enabled', True)
    
    async def validate_results(
        self, 
        results: List[RankedResult],
        query: Optional[str] = None
    ) -> ValidationReport:
        """
        Validate search results by cross-referencing information across sources.
        
        Args:
            results: List of search results to validate
            query: Original search query for context
            
        Returns:
            Validation report with consistency analysis
        """
        if not self.enabled or not results:
            return ValidationReport(
                overall_confidence=0.5,
                validation_results=[],
                consistency_checks=[],
                source_agreement_score=0.5,
                conflicting_information=[],
                supporting_evidence=[]
            )
        
        try:
            logger.info(f"Validating {len(results)} search results")
            
            # Step 1: Extract claims from all results
            all_claims = []
            for result in results:
                claims = self.claim_extractor.extract_claims(
                    result.content_snippet, 
                    result.chunk_id
                )
                all_claims.extend(claims)
            
            logger.info(f"Extracted {len(all_claims)} claims for validation")
            
            # Step 2: Analyze consistency between claims
            consistency_checks = await self.consistency_analyzer.analyze_consistency(all_claims)
            
            # Step 3: Generate validation results for each original result
            validation_results = self._generate_validation_results(results, consistency_checks)
            
            # Step 4: Calculate overall metrics
            overall_confidence = self._calculate_overall_confidence(validation_results, consistency_checks)
            source_agreement = self._calculate_source_agreement(consistency_checks)
            
            # Step 5: Identify conflicts and supporting evidence
            conflicts = self._identify_conflicts(consistency_checks)
            support = self._identify_supporting_evidence(consistency_checks)
            
            validation_report = ValidationReport(
                overall_confidence=overall_confidence,
                validation_results=validation_results,
                consistency_checks=len(consistency_checks),
                source_agreement_score=source_agreement,
                conflicting_information=conflicts,
                supporting_evidence=support
            )
            
            logger.info(f"Validation completed. Overall confidence: {overall_confidence:.3f}")
            return validation_report
            
        except Exception as e:
            logger.error(f"Error in result validation: {e}")
            # Return default validation report
            return ValidationReport(
                overall_confidence=0.5,
                validation_results=[],
                consistency_checks=0,
                source_agreement_score=0.5,
                conflicting_information=[],
                supporting_evidence=[]
            )
    
    def _generate_validation_results(
        self, 
        results: List[RankedResult], 
        consistency_checks: List[ConsistencyCheck]
    ) -> List[ValidationResult]:
        """Generate validation results for each search result."""
        validation_results = []
        
        for result in results:
            # Find consistency checks involving this result
            related_checks = [
                check for check in consistency_checks
                if (check.claim1.source_chunk_id == result.chunk_id or 
                    check.claim2.source_chunk_id == result.chunk_id)
            ]
            
            # Calculate confidence based on consistency checks
            confidence_scores = []
            supporting_count = 0
            contradicting_count = 0
            
            for check in related_checks:
                confidence_scores.append(check.confidence)
                if check.consistency_type == "supporting":
                    supporting_count += 1
                elif check.consistency_type == "contradicting":
                    contradicting_count += 1
            
            # Calculate overall confidence for this result
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
            else:
                avg_confidence = 0.5  # Neutral when no cross-references
            
            # Adjust based on support vs contradiction ratio
            if supporting_count + contradicting_count > 0:
                support_ratio = supporting_count / (supporting_count + contradicting_count)
                confidence_adjustment = (support_ratio - 0.5) * 0.2  # Max ±0.1 adjustment
                avg_confidence += confidence_adjustment
            
            validation_result = ValidationResult(
                chunk_id=result.chunk_id,
                confidence_score=max(0.1, min(0.9, avg_confidence)),
                supporting_sources=supporting_count,
                contradicting_sources=contradicting_count,
                cross_references=len(related_checks),
                validation_notes=f"Cross-referenced with {len(related_checks)} other sources"
            )
            
            validation_results.append(validation_result)
        
        return validation_results
    
    def _calculate_overall_confidence(
        self, 
        validation_results: List[ValidationResult], 
        consistency_checks: List[ConsistencyCheck]
    ) -> float:
        """Calculate overall confidence score for the result set."""
        if not validation_results:
            return 0.5
        
        # Average individual confidence scores
        individual_confidences = [vr.confidence_score for vr in validation_results]
        avg_individual = sum(individual_confidences) / len(individual_confidences)
        
        # Bonus for having cross-references
        cross_ref_bonus = 0.0
        if consistency_checks:
            cross_ref_bonus = min(len(consistency_checks) * 0.01, 0.1)  # Max 0.1 bonus
        
        overall_confidence = avg_individual + cross_ref_bonus
        return max(0.1, min(0.9, overall_confidence))
    
    def _calculate_source_agreement(self, consistency_checks: List[ConsistencyCheck]) -> float:
        """Calculate how much sources agree with each other."""
        if not consistency_checks:
            return 0.5
        
        supporting_count = sum(1 for check in consistency_checks if check.consistency_type == "supporting")
        contradicting_count = sum(1 for check in consistency_checks if check.consistency_type == "contradicting")
        total_checks = len(consistency_checks)
        
        if total_checks == 0:
            return 0.5
        
        # Agreement score based on ratio of supporting vs contradicting
        if supporting_count + contradicting_count == 0:
            return 0.5  # All neutral
        
        agreement_ratio = supporting_count / (supporting_count + contradicting_count)
        return agreement_ratio
    
    def _identify_conflicts(self, consistency_checks: List[ConsistencyCheck]) -> List[str]:
        """Identify conflicting information from consistency checks."""
        conflicts = []
        
        contradicting_checks = [
            check for check in consistency_checks 
            if check.consistency_type == "contradicting" and check.confidence > 0.6
        ]
        
        for check in contradicting_checks[:5]:  # Limit to top 5 conflicts
            conflict_description = (
                f"Conflicting information found: "
                f"'{check.claim1.text[:100]}...' vs "
                f"'{check.claim2.text[:100]}...' "
                f"(confidence: {check.confidence:.2f})"
            )
            conflicts.append(conflict_description)
        
        return conflicts
    
    def _identify_supporting_evidence(self, consistency_checks: List[ConsistencyCheck]) -> List[str]:
        """Identify supporting evidence from consistency checks."""
        support = []
        
        supporting_checks = [
            check for check in consistency_checks 
            if check.consistency_type == "supporting" and check.confidence > 0.7
        ]
        
        for check in supporting_checks[:5]:  # Limit to top 5 supporting evidence
            support_description = (
                f"Supporting evidence found: "
                f"Multiple sources confirm similar information "
                f"(confidence: {check.confidence:.2f})"
            )
            support.append(support_description)
        
        return support
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation service statistics."""
        return {
            "validation_enabled": self.enabled,
            "claim_extraction_patterns": len(self.claim_extractor.fact_patterns),
            "consistency_threshold": self.consistency_analyzer.similarity_threshold,
            "supported_consistency_types": ["supporting", "contradicting", "neutral"]
        }


# Global instance
validation_service = ValidationService()
