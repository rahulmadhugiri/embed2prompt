#!/usr/bin/env python3
"""
Evaluation Script for Embedding-to-Prompt Model
Measures cosine similarity, diversity, and quality metrics for generated prompts.
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

# Import our inference engine
import sys
sys.path.append('.')
from scripts.inference import EmbeddingToPromptInference
from dotenv import load_dotenv

# Optional imports
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.rouge_score import rouge_l_sentence_level
    from nltk.tokenize import word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from rouge import Rouge
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingToPromptEvaluator:
    """Comprehensive evaluator for embedding-to-prompt models."""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to the configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize inference engine
        self.inference = EmbeddingToPromptInference(model_path, config_path)
        
        # Initialize OpenAI client
        self.openai_client = None
        if HAS_OPENAI and os.getenv('OPENAI_API_KEY'):
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize NLTK if available
        if HAS_NLTK:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
        
        logger.info("Evaluator initialized successfully")
    
    def compute_embedding_similarity(self, embedding1: np.ndarray, 
                                   embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if HAS_SKLEARN:
            return cosine_similarity([embedding1], [embedding2])[0][0]
        else:
            # Manual computation
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
    
    def compute_text_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Compute various text similarity metrics."""
        metrics = {}
        
        if HAS_NLTK:
            # BLEU score
            try:
                tokens1 = word_tokenize(text1.lower())
                tokens2 = word_tokenize(text2.lower())
                
                if tokens1 and tokens2:
                    smoothing = SmoothingFunction()
                    bleu = sentence_bleu([tokens1], tokens2, 
                                       smoothing_function=smoothing.method1)
                    metrics['bleu'] = bleu
            except Exception as e:
                logger.warning(f"BLEU computation failed: {e}")
        
        if HAS_ROUGE:
            # ROUGE score
            try:
                rouge = Rouge()
                scores = rouge.get_scores(text1, text2)
                metrics['rouge_1'] = scores[0]['rouge-1']['f']
                metrics['rouge_2'] = scores[0]['rouge-2']['f']
                metrics['rouge_l'] = scores[0]['rouge-l']['f']
            except Exception as e:
                logger.warning(f"ROUGE computation failed: {e}")
        
        # Simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if words1 and words2:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            metrics['jaccard'] = intersection / union
            metrics['word_overlap'] = intersection / min(len(words1), len(words2))
        
        return metrics
    
    def evaluate_prompt_effectiveness(self, prompt: str, target_embedding: np.ndarray,
                                    test_models: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate how effectively a prompt generates content similar to the target.
        
        Args:
            prompt: Generated prompt
            target_embedding: Target embedding
            test_models: List of models to test with
            
        Returns:
            Dictionary with effectiveness metrics
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available. Skipping effectiveness evaluation.")
            return {}
        
        if test_models is None:
            test_models = ["gpt-3.5-turbo"]
        
        results = {}
        
        for model in test_models:
            try:
                # Generate response using the prompt
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                
                generated_text = response.choices[0].message.content
                
                # Embed the generated text
                embedding_response = self.openai_client.embeddings.create(
                    input=generated_text,
                    model=self.config['openai']['embedding_model']
                )
                
                generated_embedding = np.array(embedding_response.data[0].embedding)
                
                # Compute similarity
                cosine_sim = self.compute_embedding_similarity(target_embedding, generated_embedding)
                
                results[model] = {
                    'cosine_similarity': float(cosine_sim),
                    'generated_text': generated_text,
                    'prompt_length': len(prompt.split()),
                    'response_length': len(generated_text.split())
                }
                
            except Exception as e:
                logger.error(f"Error evaluating with {model}: {e}")
                results[model] = {'error': str(e)}
        
        return results
    
    def compute_diversity_metrics(self, prompts: List[str]) -> Dict[str, float]:
        """
        Compute diversity metrics for a set of prompts.
        
        Args:
            prompts: List of generated prompts
            
        Returns:
            Dictionary with diversity metrics
        """
        if not prompts:
            return {}
        
        metrics = {}
        
        # Vocabulary diversity
        all_words = []
        for prompt in prompts:
            words = prompt.lower().split()
            all_words.extend(words)
        
        unique_words = set(all_words)
        total_words = len(all_words)
        
        metrics['vocabulary_size'] = len(unique_words)
        metrics['type_token_ratio'] = len(unique_words) / total_words if total_words > 0 else 0
        
        # Length diversity
        lengths = [len(prompt.split()) for prompt in prompts]
        metrics['avg_length'] = np.mean(lengths)
        metrics['length_std'] = np.std(lengths)
        metrics['length_range'] = max(lengths) - min(lengths)
        
        # Pairwise similarity
        similarities = []
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                text_sim = self.compute_text_similarity(prompts[i], prompts[j])
                if 'jaccard' in text_sim:
                    similarities.append(text_sim['jaccard'])
        
        if similarities:
            metrics['avg_pairwise_similarity'] = np.mean(similarities)
            metrics['similarity_std'] = np.std(similarities)
        
        # Distinct n-grams
        for n in [1, 2, 3]:
            all_ngrams = []
            for prompt in prompts:
                words = prompt.lower().split()
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                all_ngrams.extend(ngrams)
            
            unique_ngrams = set(all_ngrams)
            total_ngrams = len(all_ngrams)
            
            metrics[f'distinct_{n}grams'] = len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0
        
        return metrics
    
    def evaluate_test_set(self, test_data_path: str, 
                         num_samples: Optional[int] = None,
                         num_prompt_candidates: int = 1) -> Dict[str, Any]:
        """
        Evaluate the model on a test set.
        
        Args:
            test_data_path: Path to test data
            num_samples: Number of samples to evaluate (None for all)
            num_prompt_candidates: Number of prompt candidates per embedding
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating on test set: {test_data_path}")
        
        # Load test data
        if test_data_path.endswith('.jsonl'):
            import jsonlines
            with jsonlines.open(test_data_path, 'r') as reader:
                test_data = [item for item in reader]
        else:
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)
        
        if num_samples:
            test_data = test_data[:num_samples]
        
        logger.info(f"Evaluating {len(test_data)} samples")
        
        # Evaluation metrics
        embedding_similarities = []
        text_similarities = []
        effectiveness_scores = []
        diversity_metrics = []
        prompt_lengths = []
        
        # Process each sample
        for sample in tqdm(test_data, desc="Evaluating samples"):
            try:
                # Get target embedding and reference prompt
                target_embedding = np.array(sample['embedding'])
                reference_prompt = sample['prompt']
                
                # Generate prompt(s)
                if num_prompt_candidates == 1:
                    generated_prompts = [self.inference.generate_prompt(target_embedding)]
                else:
                    generated_prompts = self.inference.generate_similar_prompts(
                        target_embedding, num_prompt_candidates
                    )
                
                # Evaluate each generated prompt
                for generated_prompt in generated_prompts:
                    # Embedding similarity (generate embedding for the prompt)
                    if self.openai_client:
                        try:
                            # Generate response from the prompt
                            response = self.openai_client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": generated_prompt}],
                                max_tokens=500,
                                temperature=0.7
                            )
                            
                            generated_text = response.choices[0].message.content
                            
                            # Embed the generated text
                            embedding_response = self.openai_client.embeddings.create(
                                input=generated_text,
                                model=self.config['openai']['embedding_model']
                            )
                            
                            generated_embedding = np.array(embedding_response.data[0].embedding)
                            
                            # Compute embedding similarity
                            emb_sim = self.compute_embedding_similarity(target_embedding, generated_embedding)
                            embedding_similarities.append(emb_sim)
                            
                        except Exception as e:
                            logger.warning(f"Error computing embedding similarity: {e}")
                    
                    # Text similarity with reference prompt
                    text_sim = self.compute_text_similarity(generated_prompt, reference_prompt)
                    text_similarities.append(text_sim)
                    
                    # Prompt statistics
                    prompt_lengths.append(len(generated_prompt.split()))
                
                # Diversity metrics for multiple candidates
                if num_prompt_candidates > 1:
                    div_metrics = self.compute_diversity_metrics(generated_prompts)
                    diversity_metrics.append(div_metrics)
                
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue
        
        # Aggregate results
        results = {
            'num_samples': len(test_data),
            'embedding_similarity': {
                'mean': np.mean(embedding_similarities) if embedding_similarities else 0,
                'std': np.std(embedding_similarities) if embedding_similarities else 0,
                'median': np.median(embedding_similarities) if embedding_similarities else 0,
                'min': np.min(embedding_similarities) if embedding_similarities else 0,
                'max': np.max(embedding_similarities) if embedding_similarities else 0
            },
            'prompt_length': {
                'mean': np.mean(prompt_lengths) if prompt_lengths else 0,
                'std': np.std(prompt_lengths) if prompt_lengths else 0,
                'median': np.median(prompt_lengths) if prompt_lengths else 0
            }
        }
        
        # Aggregate text similarities
        if text_similarities:
            for metric in ['bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'jaccard', 'word_overlap']:
                values = [sim.get(metric, 0) for sim in text_similarities if metric in sim]
                if values:
                    results[f'text_{metric}'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values)
                    }
        
        # Aggregate diversity metrics
        if diversity_metrics:
            for metric in ['vocabulary_size', 'type_token_ratio', 'avg_pairwise_similarity', 
                          'distinct_1grams', 'distinct_2grams', 'distinct_3grams']:
                values = [div.get(metric, 0) for div in diversity_metrics if metric in div]
                if values:
                    results[f'diversity_{metric}'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values)
                    }
        
        return results
    
    def create_evaluation_report(self, results: Dict[str, Any], 
                               output_path: str = "evaluation_report.html") -> None:
        """
        Create a comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            output_path: Path to save the report
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Embedding-to-Prompt Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 20px 0; }}
                .section {{ margin: 30px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #ffffcc; }}
            </style>
        </head>
        <body>
            <h1>Embedding-to-Prompt Evaluation Report</h1>
            
            <div class="section">
                <h2>Overview</h2>
                <p>Number of samples evaluated: {results.get('num_samples', 'N/A')}</p>
                <p>Model path: {self.model_path}</p>
                <p>Configuration: {self.config_path}</p>
            </div>
            
            <div class="section">
                <h2>Embedding Similarity Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        if 'embedding_similarity' in results:
            for metric, value in results['embedding_similarity'].items():
                html_content += f"<tr><td>Embedding Similarity ({metric})</td><td>{value:.4f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Text Similarity Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Median</th></tr>
        """
        
        text_metrics = ['text_bleu', 'text_rouge_1', 'text_rouge_2', 'text_rouge_l', 
                       'text_jaccard', 'text_word_overlap']
        
        for metric in text_metrics:
            if metric in results:
                values = results[metric]
                html_content += f"""
                <tr>
                    <td>{metric.replace('text_', '').upper()}</td>
                    <td>{values['mean']:.4f}</td>
                    <td>{values['std']:.4f}</td>
                    <td>{values['median']:.4f}</td>
                </tr>
                """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Diversity Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Median</th></tr>
        """
        
        diversity_metrics = ['diversity_vocabulary_size', 'diversity_type_token_ratio', 
                           'diversity_avg_pairwise_similarity', 'diversity_distinct_1grams',
                           'diversity_distinct_2grams', 'diversity_distinct_3grams']
        
        for metric in diversity_metrics:
            if metric in results:
                values = results[metric]
                html_content += f"""
                <tr>
                    <td>{metric.replace('diversity_', '').replace('_', ' ').title()}</td>
                    <td>{values['mean']:.4f}</td>
                    <td>{values['std']:.4f}</td>
                    <td>{values['median']:.4f}</td>
                </tr>
                """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Additional Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        if 'prompt_length' in results:
            for metric, value in results['prompt_length'].items():
                html_content += f"<tr><td>Prompt Length ({metric})</td><td>{value:.2f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Generate recommendations based on results
        if 'embedding_similarity' in results:
            emb_sim = results['embedding_similarity']['mean']
            if emb_sim > 0.8:
                html_content += "<li>✅ Excellent embedding similarity - model generates prompts that produce semantically similar content</li>"
            elif emb_sim > 0.6:
                html_content += "<li>⚠️ Good embedding similarity - consider fine-tuning for better alignment</li>"
            else:
                html_content += "<li>❌ Low embedding similarity - model may need significant improvement</li>"
        
        if 'diversity_distinct_1grams' in results:
            diversity = results['diversity_distinct_1grams']['mean']
            if diversity > 0.8:
                html_content += "<li>✅ High diversity in generated prompts</li>"
            elif diversity > 0.6:
                html_content += "<li>⚠️ Moderate diversity - consider techniques to increase variation</li>"
            else:
                html_content += "<li>❌ Low diversity - prompts may be too repetitive</li>"
        
        html_content += """
                </ul>
            </div>
            
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def plot_evaluation_results(self, results: Dict[str, Any], 
                              output_dir: str = "evaluation_plots") -> None:
        """
        Create visualization plots for evaluation results.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Plot embedding similarity distribution
        if 'embedding_similarity' in results:
            plt.figure(figsize=(10, 6))
            metrics = ['mean', 'std', 'median', 'min', 'max']
            values = [results['embedding_similarity'][m] for m in metrics]
            
            plt.bar(metrics, values)
            plt.title('Embedding Similarity Metrics')
            plt.ylabel('Cosine Similarity')
            plt.ylim(0, 1)
            
            for i, v in enumerate(values):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'embedding_similarity.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot text similarity metrics
        text_metrics = ['text_bleu', 'text_rouge_1', 'text_rouge_2', 'text_rouge_l', 
                       'text_jaccard', 'text_word_overlap']
        
        available_metrics = [m for m in text_metrics if m in results]
        if available_metrics:
            plt.figure(figsize=(12, 8))
            
            means = [results[m]['mean'] for m in available_metrics]
            stds = [results[m]['std'] for m in available_metrics]
            labels = [m.replace('text_', '').upper() for m in available_metrics]
            
            x = np.arange(len(labels))
            plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.title('Text Similarity Metrics')
            plt.xticks(x, labels, rotation=45)
            
            for i, (mean, std) in enumerate(zip(means, stds)):
                plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'text_similarity.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Evaluation plots saved to {output_path}")


def main():
    """Main function for command-line evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate embedding-to-prompt model")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--test-data", required=True, help="Path to test data")
    parser.add_argument("--num-samples", type=int, help="Number of samples to evaluate")
    parser.add_argument("--num-candidates", type=int, default=1, help="Number of prompt candidates")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--report-format", choices=["json", "html", "both"], default="both", 
                       help="Report format")
    parser.add_argument("--create-plots", action="store_true", help="Create visualization plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = EmbeddingToPromptEvaluator(args.model_path, args.config)
    
    # Run evaluation
    results = evaluator.evaluate_test_set(
        args.test_data,
        num_samples=args.num_samples,
        num_prompt_candidates=args.num_candidates
    )
    
    # Save results
    if args.report_format in ["json", "both"]:
        with open(output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_dir / 'evaluation_results.json'}")
    
    if args.report_format in ["html", "both"]:
        evaluator.create_evaluation_report(results, str(output_dir / "evaluation_report.html"))
    
    # Create plots
    if args.create_plots:
        evaluator.plot_evaluation_results(results, str(output_dir / "plots"))
    
    # Print summary
    logger.info("Evaluation completed!")
    logger.info(f"Embedding similarity: {results.get('embedding_similarity', {}).get('mean', 'N/A'):.4f}")
    
    if 'text_bleu' in results:
        logger.info(f"BLEU score: {results['text_bleu']['mean']:.4f}")
    
    if 'diversity_distinct_1grams' in results:
        logger.info(f"Diversity (distinct 1-grams): {results['diversity_distinct_1grams']['mean']:.4f}")


if __name__ == "__main__":
    main() 