"""
Content quality checker for the synthesis system.

Performs various quality checks on generated content including coherence analysis,
factual accuracy verification, hallucination detection, and citation coverage.
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger
from app.models.synthesis_schemas import (
    QualityCheck, Assertion, SupportingSource, Citation
)

logger = get_logger(__name__)


@dataclass
class QualityMetric:
    """Individual quality metric result."""
    name: str
    score: float
    passed: bool
    details: str
    suggestions: List[str]


class CoherenceAnalyzer:
    """Analyzes content coherence and structure."""
    
    def __init__(self):
        """Initialize coherence analyzer."""
        self.transition_words = {
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'contrast': ['however', 'nevertheless', 'on the other hand', 'conversely', 'yet'],
            'causation': ['therefore', 'consequently', 'as a result', 'thus', 'hence'],
            'sequence': ['first', 'second', 'next', 'finally', 'subsequently'],
            'emphasis': ['indeed', 'certainly', 'notably', 'importantly', 'significantly']
        }
        
    def analyze_coherence(self, content: str, sections: List[Dict[str, Any]]) -> QualityMetric:
        """Analyze content coherence."""
        try:
            score = 0.0
            issues = []
            suggestions = []
            
            # Check paragraph structure
            paragraphs = content.split('\n\n')
            if len(paragraphs) < 2:
                issues.append("Content lacks paragraph structure")
                suggestions.append("Break content into multiple paragraphs")
                score -= 0.2
            
            # Check for logical flow
            transition_score = self._analyze_transitions(content)
            score += transition_score * 0.3
            
            # Check section coherence
            section_score = self._analyze_section_coherence(sections)
            score += section_score * 0.4
            
            # Check sentence variety
            sentence_score = self._analyze_sentence_structure(content)
            score += sentence_score * 0.3
            
            # Normalize score
            score = max(0.0, min(1.0, score + 0.5))  # Base score of 0.5
            
            # Determine if passed
            passed = score >= 0.7 and len(issues) == 0
            
            details = f"Coherence score: {score:.2f}. "
            if issues:
                details += f"Issues: {'; '.join(issues)}. "
            
            return QualityMetric(
                name="coherence",
                score=score,
                passed=passed,
                details=details,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.warning(f"Error in coherence analysis: {e}")
            return QualityMetric(
                name="coherence",
                score=0.5,
                passed=False,
                details=f"Analysis error: {e}",
                suggestions=["Review content manually"]
            )
    
    def _analyze_transitions(self, content: str) -> float:
        """Analyze use of transition words and phrases."""
        sentences = re.split(r'[.!?]+', content)
        transition_count = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for category, words in self.transition_words.items():
                if any(word in sentence_lower for word in words):
                    transition_count += 1
                    break
        
        # Score based on transition density
        if len(sentences) > 0:
            transition_ratio = transition_count / len(sentences)
            return min(1.0, transition_ratio * 3)  # Optimal ratio around 0.33
        
        return 0.0
    
    def _analyze_section_coherence(self, sections: List[Dict[str, Any]]) -> float:
        """Analyze coherence between sections."""
        if len(sections) < 2:
            return 0.8  # Single section is coherent by default
        
        coherence_score = 0.0
        
        # Check if sections have proper titles
        titled_sections = sum(1 for section in sections if section.get('title'))
        title_score = titled_sections / len(sections)
        coherence_score += title_score * 0.5
        
        # Check section length balance
        lengths = [len(section.get('content', '')) for section in sections]
        if lengths:
            length_variance = max(lengths) / min(lengths) if min(lengths) > 0 else float('inf')
            if length_variance < 3:  # Sections are reasonably balanced
                coherence_score += 0.3
        
        # Check logical ordering (basic heuristic)
        coherence_score += 0.2  # Assume sections are properly ordered
        
        return min(1.0, coherence_score)
    
    def _analyze_sentence_structure(self, content: str) -> float:
        """Analyze sentence structure variety."""
        sentences = re.split(r'[.!?]+', content)
        if not sentences:
            return 0.0
        
        # Analyze sentence lengths
        lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        if not lengths:
            return 0.0
        
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Good variety: average 15-25 words, reasonable variance
        length_score = 1.0
        if avg_length < 10 or avg_length > 30:
            length_score -= 0.3
        if length_variance < 10:  # Too uniform
            length_score -= 0.2
        
        return max(0.0, length_score)


class FactualAccuracyChecker:
    """Checks factual accuracy of generated content against sources."""
    
    def __init__(self):
        """Initialize factual accuracy checker."""
        pass
    
    def check_accuracy(
        self, 
        content: str, 
        assertions: List[Assertion],
        sources: List[Citation]
    ) -> QualityMetric:
        """Check factual accuracy of content."""
        try:
            score = 0.0
            issues = []
            suggestions = []
            
            # Check assertion support
            supported_assertions = sum(1 for assertion in assertions 
                                     if len(assertion.supporting_sources) >= settings.assertion_min_sources)
            
            if assertions:
                support_ratio = supported_assertions / len(assertions)
                score += support_ratio * 0.6
                
                if support_ratio < 0.8:
                    issues.append(f"Only {support_ratio:.1%} of assertions are well-supported")
                    suggestions.append("Add more supporting sources for claims")
            else:
                score += 0.3  # No assertions to check
            
            # Check for conflict flags
            conflicted_assertions = sum(1 for assertion in assertions if assertion.conflict_flag)
            if assertions:
                conflict_ratio = conflicted_assertions / len(assertions)
                score -= conflict_ratio * 0.3
                
                if conflict_ratio > 0.1:
                    issues.append(f"{conflict_ratio:.1%} of assertions have conflicts")
                    suggestions.append("Resolve or acknowledge conflicting information")
            
            # Check citation coverage
            citation_coverage = self._check_citation_coverage(content, sources)
            score += citation_coverage * 0.4
            
            if citation_coverage < 0.7:
                issues.append(f"Citation coverage only {citation_coverage:.1%}")
                suggestions.append("Add more citations to support claims")
            
            # Normalize score
            score = max(0.0, min(1.0, score))
            passed = score >= 0.7 and conflicted_assertions == 0
            
            details = f"Accuracy score: {score:.2f}. "
            if issues:
                details += f"Issues: {'; '.join(issues)}. "
            
            return QualityMetric(
                name="factual_accuracy",
                score=score,
                passed=passed,
                details=details,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.warning(f"Error in factual accuracy check: {e}")
            return QualityMetric(
                name="factual_accuracy",
                score=0.5,
                passed=False,
                details=f"Analysis error: {e}",
                suggestions=["Review sources and claims manually"]
            )
    
    def _check_citation_coverage(self, content: str, sources: List[Citation]) -> float:
        """Check what percentage of claims have citations."""
        # Simple heuristic: count sentences with citations vs without
        sentences = re.split(r'[.!?]+', content)
        
        cited_sentences = 0
        total_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            total_sentences += 1
            
            # Look for citation patterns
            if re.search(r'\([^)]*\d{4}[^)]*\)', sentence):  # (Author, Year)
                cited_sentences += 1
            elif re.search(r'\[[^\]]*\]', sentence):  # [1] or [Author]
                cited_sentences += 1
            elif any(source.in_text_citation.lower() in sentence.lower() 
                    for source in sources):
                cited_sentences += 1
        
        if total_sentences > 0:
            return cited_sentences / total_sentences
        
        return 1.0  # No sentences to check


class HallucinationDetector:
    """Detects potential hallucinations in generated content."""
    
    def __init__(self):
        """Initialize hallucination detector."""
        self.suspicious_patterns = [
            r'\b(?:definitely|certainly|always|never|all|none|every)\b',  # Absolute terms
            r'\brecent studies show\b',  # Vague references
            r'\bexperts agree\b',  # Vague authorities
            r'\bit is well known\b',  # Unsupported claims
            r'\bobviously\b',  # Assumptions
            r'\bclearly\b',  # Unsupported clarity claims
        ]
    
    def detect_hallucinations(
        self, 
        content: str, 
        assertions: List[Assertion],
        sources: List[Citation]
    ) -> QualityMetric:
        """Detect potential hallucinations in content."""
        try:
            hallucination_flags = []
            suggestions = []
            
            # Check for suspicious language patterns
            for pattern in self.suspicious_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    hallucination_flags.append(f"Suspicious language: {', '.join(set(matches))}")
            
            # Check for unsupported specific claims
            specific_claims = self._find_specific_claims(content)
            for claim in specific_claims:
                if not self._is_claim_supported(claim, assertions, sources):
                    hallucination_flags.append(f"Unsupported specific claim: {claim[:100]}...")
            
            # Check for contradictions between assertions
            contradictions = self._find_contradictions(assertions)
            if contradictions:
                hallucination_flags.extend(contradictions)
            
            # Calculate score (lower is better for hallucinations)
            score = max(0.0, 1.0 - len(hallucination_flags) * 0.2)
            passed = len(hallucination_flags) == 0
            
            if hallucination_flags:
                suggestions.extend([
                    "Review flagged content for accuracy",
                    "Add supporting sources for specific claims",
                    "Avoid absolute language without evidence"
                ])
            
            details = f"Hallucination check score: {score:.2f}. "
            if hallucination_flags:
                details += f"Flags: {len(hallucination_flags)} potential issues found. "
            else:
                details += "No hallucination flags detected. "
            
            return QualityMetric(
                name="hallucination_detection",
                score=score,
                passed=passed,
                details=details,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.warning(f"Error in hallucination detection: {e}")
            return QualityMetric(
                name="hallucination_detection",
                score=0.5,
                passed=False,
                details=f"Analysis error: {e}",
                suggestions=["Review content manually for accuracy"]
            )
    
    def _find_specific_claims(self, content: str) -> List[str]:
        """Find specific factual claims that should be supported."""
        # Simple patterns for specific claims
        patterns = [
            r'(\d+(?:\.\d+)?%\s+of\s+[^.]+)',  # Percentage statistics
            r'(\$\d+(?:,\d{3})*(?:\.\d{2})?\s+[^.]+)',  # Dollar amounts
            r'(\d+(?:,\d{3})*\s+(?:people|users|cases|instances)[^.]+)',  # Counts
            r'(in \d{4}[^.]+)',  # Year-specific claims
            r'(according to [^,]+,[^.]+)',  # Attribution claims
        ]
        
        claims = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            claims.extend(matches)
        
        return claims[:10]  # Limit to avoid too many checks
    
    def _is_claim_supported(
        self, 
        claim: str, 
        assertions: List[Assertion],
        sources: List[Citation]
    ) -> bool:
        """Check if a specific claim is supported by sources."""
        claim_lower = claim.lower()
        
        # Check if claim appears in any assertion with supporting sources
        for assertion in assertions:
            if claim_lower in assertion.text.lower():
                return len(assertion.supporting_sources) > 0
        
        # Check if claim appears in source content (simple heuristic)
        # This would need access to source content, which we don't have here
        # In a real implementation, you would check against retrieved documents
        
        return False  # Conservative: assume unsupported if not found in assertions
    
    def _find_contradictions(self, assertions: List[Assertion]) -> List[str]:
        """Find contradictions between assertions."""
        contradictions = []
        
        for assertion in assertions:
            if assertion.conflict_flag and assertion.consensus_score < 0.5:
                contradictions.append(
                    f"Conflicting assertion with low consensus: {assertion.text[:100]}..."
                )
        
        return contradictions


class ContentQualityChecker:
    """
    Main quality checker that orchestrates all quality checks.
    """
    
    def __init__(self):
        """Initialize content quality checker."""
        self.coherence_analyzer = CoherenceAnalyzer()
        self.accuracy_checker = FactualAccuracyChecker()
        self.hallucination_detector = HallucinationDetector()
        
    async def check_quality(
        self,
        content: str,
        sections: List[Dict[str, Any]],
        assertions: List[Assertion],
        sources: List[Citation],
        requirements: Optional[Dict[str, Any]] = None
    ) -> QualityCheck:
        """
        Perform comprehensive quality check on generated content.
        
        Args:
            content: Generated content text
            sections: Content sections
            assertions: Extracted assertions
            sources: Source citations
            requirements: Optional quality requirements
            
        Returns:
            Comprehensive quality check results
        """
        try:
            logger.info("Performing content quality check")
            
            # Run all quality checks
            coherence_check = self.coherence_analyzer.analyze_coherence(content, sections)
            accuracy_check = self.accuracy_checker.check_accuracy(content, assertions, sources)
            hallucination_check = self.hallucination_detector.detect_hallucinations(
                content, assertions, sources
            )
            
            # Basic structural checks
            length_check = self._check_length(content, requirements)
            structure_check = self._check_structure(content, sections)
            citation_coverage = self._calculate_citation_coverage(content, sources)
            
            # Calculate overall scores
            coherence_score = coherence_check.score
            factual_accuracy_score = accuracy_check.score
            hallucination_score = hallucination_check.score
            
            # Collect all hallucination flags
            hallucination_flags = []
            if not hallucination_check.passed:
                hallucination_flags = [hallucination_check.details]
            
            # Determine overall quality
            overall_score = (
                coherence_score * 0.3 +
                factual_accuracy_score * 0.4 +
                hallucination_score * 0.3
            )
            
            if overall_score >= 0.8:
                overall_quality = "High"
            elif overall_score >= 0.6:
                overall_quality = "Medium"
            else:
                overall_quality = "Low"
            
            quality_check = QualityCheck(
                coherence_score=coherence_score,
                factual_accuracy_score=factual_accuracy_score,
                citation_coverage=citation_coverage,
                hallucination_flags=hallucination_flags,
                length_check=length_check,
                structure_check=structure_check,
                overall_quality=overall_quality
            )
            
            logger.info(f"Quality check completed: {overall_quality} quality ({overall_score:.2f})")
            return quality_check
            
        except Exception as e:
            logger.error(f"Error in quality check: {e}")
            # Return default quality check with error
            return QualityCheck(
                coherence_score=0.5,
                factual_accuracy_score=0.5,
                citation_coverage=0.5,
                hallucination_flags=[f"Quality check error: {e}"],
                length_check=False,
                structure_check=False,
                overall_quality="Low"
            )
    
    def _check_length(self, content: str, requirements: Optional[Dict[str, Any]]) -> bool:
        """Check if content meets length requirements."""
        if not requirements:
            return True  # No requirements specified
        
        min_length = requirements.get('min_length', 0)
        max_length = requirements.get('max_length', float('inf'))
        
        content_length = len(content)
        return min_length <= content_length <= max_length
    
    def _check_structure(self, content: str, sections: List[Dict[str, Any]]) -> bool:
        """Check if content has proper structure."""
        # Basic structure checks
        has_headings = bool(re.search(r'^#+\s+', content, re.MULTILINE))
        has_paragraphs = len(content.split('\n\n')) > 1
        has_sections = len(sections) > 0
        
        return has_headings and has_paragraphs and has_sections
    
    def _calculate_citation_coverage(self, content: str, sources: List[Citation]) -> float:
        """Calculate citation coverage percentage."""
        if not sources:
            return 0.0
        
        # Count citation references in content
        citation_count = 0
        for source in sources:
            if source.in_text_citation in content:
                citation_count += 1
        
        return citation_count / len(sources) if sources else 0.0


# Global quality checker instance
quality_checker = ContentQualityChecker()
