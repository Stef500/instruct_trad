"""
Unit tests for the SampleSelector class.
"""
import pytest
import random
from unittest.mock import patch

from src.medical_dataset_processor.models.core import Sample
from src.medical_dataset_processor.processors.sample_selector import SampleSelector


class TestSampleSelector:
    """Test cases for SampleSelector class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create test samples
        self.test_samples = []
        for i in range(150):  # Create enough samples for testing
            sample = Sample(
                id=f"sample_{i}",
                content={"text": f"Sample text {i}", "label": f"label_{i}"},
                source_dataset=f"dataset_{i % 3}",  # 3 different datasets
                original_text=f"This is sample text number {i}"
            )
            self.test_samples.append(sample)
        
        # Create selector with fixed seed for reproducible tests
        self.selector = SampleSelector(random_seed=42)
    
    def test_init_with_seed(self):
        """Test initialization with random seed."""
        selector = SampleSelector(random_seed=123)
        assert selector.random_seed == 123
    
    def test_init_without_seed(self):
        """Test initialization without random seed."""
        selector = SampleSelector()
        assert selector.random_seed is None
    
    def test_select_for_translation_default_count(self):
        """Test selecting samples for translation with default count."""
        selected = self.selector.select_for_translation(self.test_samples)
        
        assert len(selected) == 50  # Default count
        assert all(isinstance(sample, Sample) for sample in selected)
        
        # Check that all selected samples are from the original list
        original_ids = {sample.id for sample in self.test_samples}
        selected_ids = {sample.id for sample in selected}
        assert selected_ids.issubset(original_ids)
    
    def test_select_for_translation_custom_count(self):
        """Test selecting samples for translation with custom count."""
        count = 25
        selected = self.selector.select_for_translation(self.test_samples, count)
        
        assert len(selected) == count
        assert all(isinstance(sample, Sample) for sample in selected)
    
    def test_select_for_translation_insufficient_samples(self):
        """Test error when requesting more samples than available."""
        small_sample_list = self.test_samples[:10]
        
        with pytest.raises(ValueError, match="Not enough samples available"):
            self.selector.select_for_translation(small_sample_list, 20)
    
    def test_select_for_translation_invalid_count(self):
        """Test error with invalid count values."""
        with pytest.raises(ValueError, match="Count must be positive"):
            self.selector.select_for_translation(self.test_samples, 0)
        
        with pytest.raises(ValueError, match="Count must be positive"):
            self.selector.select_for_translation(self.test_samples, -5)
    
    def test_select_for_generation_default_count(self):
        """Test selecting samples for generation with default count."""
        selected = self.selector.select_for_generation(self.test_samples)
        
        assert len(selected) == 50  # Default count
        assert all(isinstance(sample, Sample) for sample in selected)
    
    def test_select_for_generation_with_exclusions(self):
        """Test selecting samples for generation with exclusions."""
        # First select some samples to exclude
        exclude_samples = self.test_samples[:20]
        
        selected = self.selector.select_for_generation(
            self.test_samples, count=30, exclude=exclude_samples
        )
        
        assert len(selected) == 30
        
        # Verify no overlap with excluded samples
        excluded_ids = {sample.id for sample in exclude_samples}
        selected_ids = {sample.id for sample in selected}
        assert not excluded_ids.intersection(selected_ids)
    
    def test_select_for_generation_insufficient_after_exclusions(self):
        """Test error when not enough samples remain after exclusions."""
        # Exclude most samples, leaving only a few
        exclude_samples = self.test_samples[:140]
        
        with pytest.raises(ValueError, match="Not enough samples available after exclusions"):
            self.selector.select_for_generation(
                self.test_samples, count=20, exclude=exclude_samples
            )
    
    def test_select_for_generation_invalid_count(self):
        """Test error with invalid count values for generation."""
        with pytest.raises(ValueError, match="Count must be positive"):
            self.selector.select_for_generation(self.test_samples, 0)
        
        with pytest.raises(ValueError, match="Count must be positive"):
            self.selector.select_for_generation(self.test_samples, -3)
    
    def test_select_samples_by_dataset(self):
        """Test selecting samples for both translation and generation."""
        translation_samples, generation_samples = self.selector.select_samples_by_dataset(
            self.test_samples, translation_count=30, generation_count=40
        )
        
        assert len(translation_samples) == 30
        assert len(generation_samples) == 40
        
        # Verify no overlap
        translation_ids = {sample.id for sample in translation_samples}
        generation_ids = {sample.id for sample in generation_samples}
        assert not translation_ids.intersection(generation_ids)
    
    def test_select_samples_by_dataset_insufficient_total(self):
        """Test error when not enough samples for both operations."""
        small_sample_list = self.test_samples[:80]
        
        with pytest.raises(ValueError, match="Not enough samples for both operations"):
            self.selector.select_samples_by_dataset(
                small_sample_list, translation_count=50, generation_count=50
            )
    
    def test_validate_no_overlap_success(self):
        """Test validation when there's no overlap between samples."""
        translation_samples = self.test_samples[:25]
        generation_samples = self.test_samples[25:50]
        
        result = self.selector.validate_no_overlap(translation_samples, generation_samples)
        assert result is True
    
    def test_validate_no_overlap_failure(self):
        """Test validation when there's overlap between samples."""
        translation_samples = self.test_samples[:30]
        generation_samples = self.test_samples[20:50]  # Overlap from 20-30
        
        result = self.selector.validate_no_overlap(translation_samples, generation_samples)
        assert result is False
    
    def test_get_selection_stats(self):
        """Test getting selection statistics."""
        translation_samples = self.test_samples[:25]
        generation_samples = self.test_samples[25:50]
        
        stats = self.selector.get_selection_stats(
            self.test_samples, translation_samples, generation_samples
        )
        
        assert stats["total_samples"] == len(self.test_samples)
        assert stats["translation_samples"] == 25
        assert stats["generation_samples"] == 25
        assert stats["unused_samples"] == len(self.test_samples) - 50
        assert stats["has_overlap"] is False
        
        # Check dataset distribution
        assert "dataset_distribution" in stats
        assert "translation_dataset_distribution" in stats
        assert "generation_dataset_distribution" in stats
    
    def test_get_selection_stats_with_overlap(self):
        """Test getting selection statistics when there's overlap."""
        translation_samples = self.test_samples[:30]
        generation_samples = self.test_samples[20:50]  # Overlap from 20-30
        
        stats = self.selector.get_selection_stats(
            self.test_samples, translation_samples, generation_samples
        )
        
        assert stats["has_overlap"] is True
    
    def test_reproducible_selection_with_seed(self):
        """Test that selection is reproducible with the same seed."""
        selector1 = SampleSelector(random_seed=123)
        selector2 = SampleSelector(random_seed=123)
        
        selected1 = selector1.select_for_translation(self.test_samples, 20)
        selected2 = selector2.select_for_translation(self.test_samples, 20)
        
        # Should get the same samples in the same order
        assert len(selected1) == len(selected2)
        for s1, s2 in zip(selected1, selected2):
            assert s1.id == s2.id
    
    def test_different_selection_with_different_seeds(self):
        """Test that selection differs with different seeds."""
        selector1 = SampleSelector(random_seed=123)
        selector2 = SampleSelector(random_seed=456)
        
        selected1 = selector1.select_for_translation(self.test_samples, 20)
        selected2 = selector2.select_for_translation(self.test_samples, 20)
        
        # Should get different samples (very unlikely to be identical)
        selected1_ids = {sample.id for sample in selected1}
        selected2_ids = {sample.id for sample in selected2}
        assert selected1_ids != selected2_ids
    
    def test_empty_exclude_list(self):
        """Test generation selection with empty exclude list."""
        selected = self.selector.select_for_generation(
            self.test_samples, count=20, exclude=[]
        )
        
        assert len(selected) == 20
        assert all(isinstance(sample, Sample) for sample in selected)
    
    def test_none_exclude_list(self):
        """Test generation selection with None exclude list."""
        selected = self.selector.select_for_generation(
            self.test_samples, count=20, exclude=None
        )
        
        assert len(selected) == 20
        assert all(isinstance(sample, Sample) for sample in selected)
    
    def test_select_all_available_samples(self):
        """Test selecting exactly the number of available samples."""
        small_sample_list = self.test_samples[:50]
        
        selected = self.selector.select_for_translation(small_sample_list, 50)
        assert len(selected) == 50
        
        # All samples should be selected (order may differ)
        original_ids = {sample.id for sample in small_sample_list}
        selected_ids = {sample.id for sample in selected}
        assert original_ids == selected_ids
    
    @patch('src.medical_dataset_processor.processors.sample_selector.logger')
    def test_logging_calls(self, mock_logger):
        """Test that appropriate logging calls are made."""
        self.selector.select_for_translation(self.test_samples, 30)
        mock_logger.info.assert_called()
        
        self.selector.select_for_generation(self.test_samples, 25)
        mock_logger.info.assert_called()
    
    def test_dataset_distribution_in_stats(self):
        """Test that dataset distribution is correctly calculated in stats."""
        # Use a smaller, controlled set of samples
        controlled_samples = []
        for i in range(60):
            sample = Sample(
                id=f"sample_{i}",
                content={"text": f"Sample text {i}"},
                source_dataset=f"dataset_{i % 2}",  # Only 2 datasets
                original_text=f"This is sample text number {i}"
            )
            controlled_samples.append(sample)
        
        translation_samples = controlled_samples[:20]
        generation_samples = controlled_samples[20:40]
        
        stats = self.selector.get_selection_stats(
            controlled_samples, translation_samples, generation_samples
        )
        
        # Check that dataset distribution makes sense
        total_dist = stats["dataset_distribution"]
        assert "dataset_0" in total_dist
        assert "dataset_1" in total_dist
        assert total_dist["dataset_0"] == 30  # Half of 60
        assert total_dist["dataset_1"] == 30  # Half of 60