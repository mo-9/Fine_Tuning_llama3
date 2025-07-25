
import unittest
import tempfile
import os
from unittest.mock import Mock, patch

# Import pipeline components
from data_collection.data_collector import DataCollector
from data_processing.data_cleaner import DataCleaner
from data_processing.data_storage import DataStorage
from fine_tuning.qa_generator import QAGenerator
from evaluation.benchmark_generator import BenchmarkGenerator
from deployment.model_registry import ModelRegistry
from orchestration.pipeline_orchestrator import PipelineOrchestrator

class TestDataCollection(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.collector = DataCollector(storage_path=os.path.join(self.temp_dir, "test.db"))
    
    def test_data_collector_initialization(self):
        """Test that DataCollector initializes correctly."""
        self.assertIsNotNone(self.collector.web_scraper)
        self.assertIsNotNone(self.collector.pdf_extractor)
        self.assertIsNotNone(self.collector.data_cleaner)
        self.assertIsNotNone(self.collector.data_storage)

class TestDataProcessing(unittest.TestCase):
    """Test data processing components."""
    
    def setUp(self):
        self.cleaner = DataCleaner()
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        dirty_text = "  This   is    a   test   text!!!  "
        cleaned = self.cleaner.clean_text(dirty_text)
        self.assertEqual(cleaned, "This is a test text!")
    
    def test_quality_filter(self):
        """Test quality filtering."""
        good_text = "This is a good quality text with sufficient length and common English words."
        bad_text = "xyz"
        
        self.assertTrue(self.cleaner.quality_filter(good_text))
        self.assertFalse(self.cleaner.quality_filter(bad_text))
    
    def test_duplicate_detection(self):
        """Test duplicate detection."""
        text1 = "This is a test text"
        text2 = "This is a test text"  # Same as text1
        text3 = "This is a different text"
        
        # First occurrence should not be duplicate
        self.assertFalse(self.cleaner.is_duplicate(text1))
        # Second occurrence should be duplicate
        self.assertTrue(self.cleaner.is_duplicate(text2))
        # Different text should not be duplicate
        self.assertFalse(self.cleaner.is_duplicate(text3))

class TestDataStorage(unittest.TestCase):
    """Test data storage functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = DataStorage(os.path.join(self.temp_dir, "test.db"))
    
    def test_document_storage_and_retrieval(self):
        """Test storing and retrieving documents."""
        test_docs = [
            {
                "url": "http://example.com",
                "title": "Test Document",
                "content": "This is test content",
                "source": "test",
                "timestamp": 1234567890
            }
        ]
        
        # Store documents
        doc_ids = self.storage.store_documents(test_docs)
        self.assertEqual(len(doc_ids), 1)
        
        # Retrieve documents
        retrieved_docs = self.storage.get_documents()
        self.assertEqual(len(retrieved_docs), 1)
        self.assertEqual(retrieved_docs[0]["title"], "Test Document")

class TestQAGeneration(unittest.TestCase):
    """Test Q&A generation functionality."""
    
    def setUp(self):
        # Mock the QA pipeline to avoid loading actual models in tests
        with patch('fine_tuning.qa_generator.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            self.qa_generator = QAGenerator()
    
    def test_qa_pair_generation(self):
        """Test Q&A pair generation from context."""
        context = "Electric vehicles use rechargeable batteries. They are environmentally friendly."
        
        # Mock the QA pipeline response
        self.qa_generator.qa_pipeline.return_value = {"answer": "rechargeable batteries"}
        
        qa_pairs = self.qa_generator.generate_qa_pairs(context, num_questions=1)
        
        self.assertEqual(len(qa_pairs), 1)
        self.assertIn("question", qa_pairs[0])
        self.assertIn("answer", qa_pairs[0])
        self.assertIn("context", qa_pairs[0])

class TestBenchmarkGeneration(unittest.TestCase):
    """Test benchmark generation functionality."""
    
    def setUp(self):
        self.benchmark_gen = BenchmarkGenerator()
    
    def test_benchmark_dataset_generation(self):
        """Test benchmark dataset generation."""
        qa_pairs = [
            {"question": "What is EV?", "answer": "Electric Vehicle", "context": "EVs are cars..."},
            {"question": "How do EVs work?", "answer": "Using batteries", "context": "EVs use..."}
        ]
        
        benchmark_dataset = self.benchmark_gen.generate_benchmark_dataset(qa_pairs, num_samples=2)
        
        self.assertEqual(len(benchmark_dataset), 2)
        self.assertIn("question", benchmark_dataset[0])
        self.assertIn("ground_truth_answer", benchmark_dataset[0])

class TestModelRegistry(unittest.TestCase):
    """Test model registry functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.temp_dir)
    
    def test_model_registration(self):
        """Test model registration and retrieval."""
        model_id = self.registry.register_model(
            model_name="test_model",
            version="v1.0",
            model_path="/path/to/model",
            metadata={"test": "metadata"}
        )
        
        self.assertEqual(model_id, "test_model:v1.0")
        
        # Retrieve the model
        model_info = self.registry.get_model("test_model", "v1.0")
        self.assertEqual(model_info["model_path"], "/path/to/model")
        self.assertEqual(model_info["metadata"]["test"], "metadata")
    
    def test_model_listing(self):
        """Test listing all models."""
        # Register multiple models
        self.registry.register_model("model1", "v1.0", "/path1")
        self.registry.register_model("model2", "v1.0", "/path2")
        
        models = self.registry.list_models()
        self.assertEqual(len(models), 2)

class TestPipelineOrchestrator(unittest.TestCase):
    """Test pipeline orchestration functionality."""
    
    def setUp(self):
        self.orchestrator = PipelineOrchestrator()
    
    def test_config_loading(self):
        """Test configuration loading."""
        self.assertIsNotNone(self.orchestrator.config)
        self.assertIn("data_collection", self.orchestrator.config)
        self.assertIn("training", self.orchestrator.config)
        self.assertIn("evaluation", self.orchestrator.config)
        self.assertIn("deployment", self.orchestrator.config)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    @patch('fine_tuning.qa_generator.pipeline')
    def test_end_to_end_pipeline_simulation(self, mock_pipeline):
        """Test a simplified end-to-end pipeline execution."""
        # Mock external dependencies
        mock_pipeline.return_value = Mock()
        mock_pipeline.return_value.return_value = {"answer": "test answer"}
        
        # Create test data
        test_docs = [
            {
                "url": "http://example.com",
                "title": "Test EV Document",
                "content": "Electric vehicles are powered by rechargeable batteries. They produce zero emissions.",
                "source": "test",
                "timestamp": 1234567890
            }
        ]
        
        # Test data storage
        storage = DataStorage(os.path.join(self.temp_dir, "test.db"))
        doc_ids = storage.store_documents(test_docs)
        self.assertEqual(len(doc_ids), 1)
        
        # Test QA generation
        qa_generator = QAGenerator()
        qa_pairs = qa_generator.generate_qa_from_documents(test_docs, num_questions_per_doc=1)
        self.assertGreater(len(qa_pairs), 0)
        
        # Test benchmark generation
        benchmark_gen = BenchmarkGenerator()
        benchmark_dataset = benchmark_gen.generate_benchmark_dataset(qa_pairs)
        self.assertGreater(len(benchmark_dataset), 0)
        
        # Test model registry
        registry = ModelRegistry(os.path.join(self.temp_dir, "registry"))
        model_id = registry.register_model("test_model", "v1.0", "/test/path")
        self.assertEqual(model_id, "test_model:v1.0")

if __name__ == "__main__":
    # Set up test environment
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests
    unittest.main(verbosity=2)

