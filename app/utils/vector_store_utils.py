"""
Utilities and helper functions for vector store operations.
Includes configuration management, monitoring, and maintenance tools.
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from app.services.vector_store_service import (
    VectorStoreService, SearchService, IndexConfig, IndexType
)
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for vector store operations."""
    timestamp: datetime
    operation: str
    duration_ms: float
    vector_count: int
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class IndexOptimizationReport:
    """Report from index optimization analysis."""
    current_index_type: str
    recommended_index_type: str
    current_performance: Dict[str, float]
    estimated_improvement: Dict[str, float]
    optimization_reason: str
    should_optimize: bool


class VectorStoreMonitor:
    """Monitor and collect metrics for vector store operations."""
    
    def __init__(self, storage_path: str):
        """Initialize monitor."""
        self.storage_path = Path(storage_path)
        self.metrics_file = self.storage_path / "metrics.jsonl"
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric."""
        try:
            with open(self.metrics_file, 'a') as f:
                metric_dict = asdict(metric)
                metric_dict['timestamp'] = metric.timestamp.isoformat()
                f.write(json.dumps(metric_dict) + '\n')
        except Exception as e:
            logger.warning(f"Failed to record metric: {e}")
    
    def get_recent_metrics(self, hours: int = 24) -> List[PerformanceMetrics]:
        """Get metrics from the last N hours."""
        if not self.metrics_file.exists():
            return []
        
        cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
        metrics = []
        
        try:
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    metric_dict = json.loads(line.strip())
                    timestamp = datetime.fromisoformat(metric_dict['timestamp'])
                    
                    if timestamp.timestamp() >= cutoff_time:
                        metric_dict['timestamp'] = timestamp
                        metrics.append(PerformanceMetrics(**metric_dict))
        except Exception as e:
            logger.warning(f"Failed to read metrics: {e}")
        
        return metrics
    
    def analyze_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze recent performance metrics."""
        metrics = self.get_recent_metrics(hours)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        # Group by operation
        by_operation = {}
        for metric in metrics:
            if metric.operation not in by_operation:
                by_operation[metric.operation] = []
            by_operation[metric.operation].append(metric)
        
        # Calculate statistics
        analysis = {}
        for operation, op_metrics in by_operation.items():
            durations = [m.duration_ms for m in op_metrics if m.success]
            success_rate = sum(1 for m in op_metrics if m.success) / len(op_metrics)
            
            analysis[operation] = {
                "count": len(op_metrics),
                "success_rate": success_rate,
                "avg_duration_ms": np.mean(durations) if durations else 0,
                "p95_duration_ms": np.percentile(durations, 95) if durations else 0,
                "errors": [m.error_message for m in op_metrics if not m.success]
            }
        
        return analysis


class IndexOptimizer:
    """Optimize FAISS index configuration based on usage patterns."""
    
    def __init__(self, monitor: VectorStoreMonitor):
        """Initialize optimizer."""
        self.monitor = monitor
    
    def analyze_index_performance(
        self, 
        vector_store: VectorStoreService,
        test_queries: List[str],
        embedding_provider
    ) -> IndexOptimizationReport:
        """Analyze current index performance and suggest optimizations."""
        
        current_config = vector_store.config
        current_metrics = vector_store.get_metrics()
        
        # Run performance test
        search_service = SearchService(vector_store, embedding_provider)
        performance = self._benchmark_search_performance(
            search_service, test_queries
        )
        
        # Determine optimal index type
        vector_count = current_metrics['total_vectors']
        dimension = current_config.dimension
        
        recommended_type = self._recommend_index_type(
            vector_count, dimension, performance
        )
        
        # Estimate improvement
        improvement = self._estimate_improvement(
            current_config.index_type, recommended_type, vector_count
        )
        
        # Generate report
        return IndexOptimizationReport(
            current_index_type=current_config.index_type.value,
            recommended_index_type=recommended_type.value,
            current_performance=performance,
            estimated_improvement=improvement,
            optimization_reason=self._get_optimization_reason(
                current_config.index_type, recommended_type, vector_count
            ),
            should_optimize=recommended_type != current_config.index_type
        )
    
    def _benchmark_search_performance(
        self, 
        search_service: SearchService, 
        test_queries: List[str]
    ) -> Dict[str, float]:
        """Benchmark search performance."""
        
        import asyncio
        
        async def run_benchmark():
            latencies = []
            
            for query in test_queries:
                start_time = time.time()
                try:
                    results = await search_service.search(query, k=10)
                    duration = (time.time() - start_time) * 1000
                    latencies.append(duration)
                except Exception as e:
                    logger.warning(f"Benchmark query failed: {e}")
            
            if latencies:
                return {
                    "avg_latency_ms": np.mean(latencies),
                    "p95_latency_ms": np.percentile(latencies, 95),
                    "p99_latency_ms": np.percentile(latencies, 99),
                    "success_rate": len(latencies) / len(test_queries)
                }
            else:
                return {"error": "All benchmark queries failed"}
        
        return asyncio.run(run_benchmark())
    
    def _recommend_index_type(
        self, 
        vector_count: int, 
        dimension: int, 
        current_performance: Dict[str, float]
    ) -> IndexType:
        """Recommend optimal index type."""
        
        # Performance-based heuristics
        avg_latency = current_performance.get("avg_latency_ms", 0)
        
        if vector_count < 1000:
            return IndexType.FLAT_IP
        elif vector_count < 10000:
            # If current latency is acceptable, stick with flat
            if avg_latency < 50:  # 50ms threshold
                return IndexType.FLAT_IP
            else:
                return IndexType.IVF_FLAT
        elif vector_count < 100000:
            return IndexType.IVF_PQ
        else:
            return IndexType.HNSW
    
    def _estimate_improvement(
        self, 
        current_type: IndexType, 
        recommended_type: IndexType, 
        vector_count: int
    ) -> Dict[str, float]:
        """Estimate performance improvement."""
        
        # Rough performance estimates based on index types
        performance_factors = {
            IndexType.FLAT_IP: {"search_speed": 1.0, "memory": 1.0, "accuracy": 1.0},
            IndexType.FLAT_L2: {"search_speed": 1.0, "memory": 1.0, "accuracy": 1.0},
            IndexType.IVF_FLAT: {"search_speed": 3.0, "memory": 0.9, "accuracy": 0.98},
            IndexType.IVF_PQ: {"search_speed": 5.0, "memory": 0.3, "accuracy": 0.95},
            IndexType.HNSW: {"search_speed": 4.0, "memory": 1.2, "accuracy": 0.99}
        }
        
        current_perf = performance_factors[current_type]
        recommended_perf = performance_factors[recommended_type]
        
        return {
            "search_speed_improvement": recommended_perf["search_speed"] / current_perf["search_speed"],
            "memory_reduction": 1 - (recommended_perf["memory"] / current_perf["memory"]),
            "accuracy_change": recommended_perf["accuracy"] / current_perf["accuracy"]
        }
    
    def _get_optimization_reason(
        self, 
        current_type: IndexType, 
        recommended_type: IndexType, 
        vector_count: int
    ) -> str:
        """Get human-readable optimization reason."""
        
        if current_type == recommended_type:
            return "Current index type is optimal"
        
        reasons = {
            (IndexType.FLAT_IP, IndexType.IVF_FLAT): f"With {vector_count} vectors, IVF can provide faster search",
            (IndexType.FLAT_IP, IndexType.IVF_PQ): f"Large collection ({vector_count} vectors) benefits from compression",
            (IndexType.FLAT_IP, IndexType.HNSW): f"HNSW provides better scalability for {vector_count} vectors",
            (IndexType.IVF_FLAT, IndexType.IVF_PQ): f"Memory usage can be reduced with quantization",
            (IndexType.IVF_FLAT, IndexType.HNSW): f"HNSW provides better performance at this scale"
        }
        
        return reasons.get(
            (current_type, recommended_type), 
            f"Switching from {current_type.value} to {recommended_type.value} for better performance"
        )


class VectorStoreUtils:
    """Utility functions for vector store management."""
    
    @staticmethod
    def estimate_storage_requirements(
        num_vectors: int, 
        dimension: int, 
        index_type: IndexType
    ) -> Dict[str, float]:
        """Estimate storage requirements for given configuration."""
        
        # Base vector storage (float32)
        base_size_mb = (num_vectors * dimension * 4) / (1024 * 1024)
        
        # Index overhead factors
        overhead_factors = {
            IndexType.FLAT_IP: 1.0,
            IndexType.FLAT_L2: 1.0,
            IndexType.IVF_FLAT: 1.1,  # Slight overhead for centroids
            IndexType.IVF_PQ: 0.25,   # Compression
            IndexType.HNSW: 1.5       # Graph structure overhead
        }
        
        index_size_mb = base_size_mb * overhead_factors[index_type]
        
        # Metadata overhead (estimated)
        metadata_size_mb = num_vectors * 0.001  # ~1KB per vector metadata
        
        return {
            "vectors_mb": base_size_mb,
            "index_mb": index_size_mb,
            "metadata_mb": metadata_size_mb,
            "total_mb": index_size_mb + metadata_size_mb
        }
    
    @staticmethod
    def validate_configuration(config: IndexConfig) -> List[str]:
        """Validate index configuration and return warnings."""
        warnings = []
        
        # Dimension checks
        if config.dimension < 1:
            warnings.append("Dimension must be positive")
        elif config.dimension > 2048:
            warnings.append("Very high dimension may impact performance")
        
        # IVF specific checks
        if config.index_type in [IndexType.IVF_FLAT, IndexType.IVF_PQ]:
            if config.nlist < 1:
                warnings.append("nlist must be positive for IVF indexes")
            elif config.nlist > 10000:
                warnings.append("Very high nlist may impact performance")
        
        # PQ specific checks
        if config.index_type == IndexType.IVF_PQ:
            if config.dimension % config.m != 0:
                warnings.append("Dimension should be divisible by m for PQ")
            if config.m > config.dimension:
                warnings.append("m should not exceed dimension")
        
        # HNSW specific checks
        if config.index_type == IndexType.HNSW:
            if config.hnsw_m < 4:
                warnings.append("HNSW M should be at least 4")
            elif config.hnsw_m > 64:
                warnings.append("Very high HNSW M may use excessive memory")
        
        return warnings
    
    @staticmethod
    def backup_vector_store(source_path: str, backup_path: str) -> bool:
        """Create backup of vector store."""
        try:
            source = Path(source_path)
            backup = Path(backup_path)
            
            if source.exists():
                if backup.exists():
                    shutil.rmtree(backup)
                shutil.copytree(source, backup)
                logger.info(f"Vector store backed up to {backup_path}")
                return True
            else:
                logger.warning(f"Source path does not exist: {source_path}")
                return False
                
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    @staticmethod
    def restore_vector_store(backup_path: str, target_path: str) -> bool:
        """Restore vector store from backup."""
        try:
            backup = Path(backup_path)
            target = Path(target_path)
            
            if backup.exists():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(backup, target)
                logger.info(f"Vector store restored from {backup_path}")
                return True
            else:
                logger.warning(f"Backup path does not exist: {backup_path}")
                return False
                
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    @staticmethod
    def cleanup_old_backups(backup_dir: str, keep_count: int = 5) -> int:
        """Clean up old backup directories, keeping only the most recent."""
        try:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                return 0
            
            # Get all backup directories
            backups = [d for d in backup_path.iterdir() if d.is_dir()]
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old backups
            removed_count = 0
            for backup in backups[keep_count:]:
                shutil.rmtree(backup)
                removed_count += 1
                logger.info(f"Removed old backup: {backup}")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0


class ConfigurationManager:
    """Manage vector store configurations."""
    
    def __init__(self, config_path: str):
        """Initialize configuration manager."""
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: IndexConfig, name: str) -> None:
        """Save configuration with a name."""
        configs = self.load_all_configs()
        configs[name] = asdict(config)
        
        with open(self.config_path, 'w') as f:
            json.dump(configs, f, indent=2)
        
        logger.info(f"Configuration '{name}' saved")
    
    def load_config(self, name: str) -> Optional[IndexConfig]:
        """Load configuration by name."""
        configs = self.load_all_configs()
        
        if name in configs:
            config_dict = configs[name]
            # Convert index_type string back to enum
            config_dict['index_type'] = IndexType(config_dict['index_type'])
            return IndexConfig(**config_dict)
        
        return None
    
    def load_all_configs(self) -> Dict[str, Any]:
        """Load all configurations."""
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load configs: {e}")
            return {}
    
    def list_configs(self) -> List[str]:
        """List all saved configuration names."""
        configs = self.load_all_configs()
        return list(configs.keys())
    
    def delete_config(self, name: str) -> bool:
        """Delete a configuration."""
        configs = self.load_all_configs()
        
        if name in configs:
            del configs[name]
            
            with open(self.config_path, 'w') as f:
                json.dump(configs, f, indent=2)
            
            logger.info(f"Configuration '{name}' deleted")
            return True
        
        return False


def create_optimal_config(
    expected_vectors: int,
    dimension: int = 768,
    search_latency_target_ms: float = 100,
    memory_limit_mb: Optional[float] = None
) -> IndexConfig:
    """Create optimal configuration based on requirements."""
    
    # Estimate requirements for different index types
    configs = []
    
    for index_type in IndexType:
        config = IndexConfig(
            index_type=index_type,
            dimension=dimension,
            auto_select=False
        )
        
        # Estimate storage
        storage = VectorStoreUtils.estimate_storage_requirements(
            expected_vectors, dimension, index_type
        )
        
        # Skip if exceeds memory limit
        if memory_limit_mb and storage['total_mb'] > memory_limit_mb:
            continue
        
        # Estimate performance (rough heuristics)
        if index_type == IndexType.FLAT_IP:
            estimated_latency = expected_vectors * 0.001  # Linear scan
        elif index_type == IndexType.IVF_FLAT:
            estimated_latency = max(10, expected_vectors * 0.0002)
        elif index_type == IndexType.IVF_PQ:
            estimated_latency = max(5, expected_vectors * 0.0001)
        elif index_type == IndexType.HNSW:
            estimated_latency = max(2, np.log2(expected_vectors) * 0.5)
        else:
            estimated_latency = 100  # Default conservative estimate
        
        if estimated_latency <= search_latency_target_ms:
            configs.append((config, storage['total_mb'], estimated_latency))
    
    if not configs:
        # Fallback to flat index
        return IndexConfig(index_type=IndexType.FLAT_IP, dimension=dimension)
    
    # Choose the most memory-efficient option that meets latency target
    configs.sort(key=lambda x: x[1])  # Sort by memory usage
    return configs[0][0]
