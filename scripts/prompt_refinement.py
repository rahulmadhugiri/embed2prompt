#!/usr/bin/env python3
"""
Prompt Refinement Module
Implements advanced prompt refinement strategies including hill climbing,
evolutionary algorithms, and selection by embedding distance.
"""

import os
import json
import yaml
import logging
import random
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from tqdm import tqdm
import copy

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
    from nltk.tokenize import word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RefinementStrategy(Enum):
    """Enumeration of available refinement strategies."""
    SIMPLE = "simple"
    HILL_CLIMBING = "hill_climbing"
    EVOLUTIONARY = "evolutionary"
    BEAM_SEARCH = "beam_search"
    SIMULATED_ANNEALING = "simulated_annealing"


@dataclass
class PromptCandidate:
    """Data class for prompt candidates."""
    prompt: str
    score: float
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RefinementConfig:
    """Configuration for prompt refinement."""
    strategy: RefinementStrategy = RefinementStrategy.SIMPLE
    max_iterations: int = 10
    population_size: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    temperature: float = 1.0
    cooling_rate: float = 0.95
    beam_width: int = 3
    diversity_penalty: float = 0.1
    convergence_threshold: float = 0.001
    max_no_improvement: int = 5


class PromptRefinement:
    """Advanced prompt refinement engine."""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        """
        Initialize the prompt refinement engine.
        
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
        
        # Initialize NLTK if needed
        if HAS_NLTK:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
        
        logger.info("Prompt refinement engine initialized successfully")
    
    def compute_prompt_score(self, prompt: str, target_embedding: np.ndarray,
                           test_model: str = "gpt-3.5-turbo") -> Tuple[float, np.ndarray]:
        """
        Compute the quality score of a prompt by generating output and comparing embeddings.
        
        Args:
            prompt: The prompt to evaluate
            target_embedding: Target embedding to compare against
            test_model: Model to use for text generation
            
        Returns:
            Tuple of (score, generated_embedding)
        """
        if not self.openai_client:
            # Fallback to simple scoring
            return random.random(), np.random.random(len(target_embedding))
        
        try:
            # Generate response using the prompt
            response = self.openai_client.chat.completions.create(
                model=test_model,
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
            
            # Compute cosine similarity as score
            score = np.dot(target_embedding, generated_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(generated_embedding)
            )
            
            return float(score), generated_embedding
            
        except Exception as e:
            logger.error(f"Error computing prompt score: {e}")
            return 0.0, np.random.random(len(target_embedding))
    
    def generate_initial_candidates(self, target_embedding: np.ndarray,
                                  num_candidates: int = 5) -> List[PromptCandidate]:
        """
        Generate initial prompt candidates.
        
        Args:
            target_embedding: Target embedding
            num_candidates: Number of candidates to generate
            
        Returns:
            List of prompt candidates
        """
        candidates = []
        
        for i in range(num_candidates):
            # Use different generation parameters for diversity
            generation_params = {
                'temperature': 0.6 + (i * 0.1),
                'top_p': 0.85 + (i * 0.02),
                'do_sample': True
            }
            
            # Generate prompt
            prompt = self.inference.generate_prompt(target_embedding, generation_params)
            
            # Compute score
            score, embedding = self.compute_prompt_score(prompt, target_embedding)
            
            candidates.append(PromptCandidate(
                prompt=prompt,
                score=score,
                embedding=embedding,
                metadata={'generation_params': generation_params}
            ))
        
        return candidates
    
    def mutate_prompt(self, prompt: str, mutation_rate: float = 0.1) -> str:
        """
        Apply mutation to a prompt.
        
        Args:
            prompt: Original prompt
            mutation_rate: Rate of mutation
            
        Returns:
            Mutated prompt
        """
        words = prompt.split()
        
        if not words:
            return prompt
        
        # Different mutation strategies
        mutations = [
            self._synonym_replacement,
            self._word_insertion,
            self._word_deletion,
            self._word_swap,
            self._sentence_restructure
        ]
        
        # Apply mutations based on rate
        mutated_words = words.copy()
        for i, word in enumerate(words):
            if random.random() < mutation_rate:
                mutation_func = random.choice(mutations)
                try:
                    mutated_words = mutation_func(mutated_words, i)
                except:
                    continue
        
        return ' '.join(mutated_words)
    
    def _synonym_replacement(self, words: List[str], index: int) -> List[str]:
        """Replace word with synonym."""
        # Simple synonym replacement (could be enhanced with WordNet)
        synonyms = {
            'create': ['generate', 'make', 'build', 'construct'],
            'explain': ['describe', 'clarify', 'elaborate', 'detail'],
            'write': ['compose', 'draft', 'author', 'pen'],
            'analyze': ['examine', 'study', 'evaluate', 'assess'],
            'compare': ['contrast', 'evaluate', 'examine', 'review']
        }
        
        word = words[index].lower()
        if word in synonyms:
            words[index] = random.choice(synonyms[word])
        
        return words
    
    def _word_insertion(self, words: List[str], index: int) -> List[str]:
        """Insert a word at random position."""
        modifiers = ['carefully', 'thoroughly', 'clearly', 'precisely', 'effectively']
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, random.choice(modifiers))
        return words
    
    def _word_deletion(self, words: List[str], index: int) -> List[str]:
        """Delete a word if the sentence is long enough."""
        if len(words) > 3:
            words.pop(index)
        return words
    
    def _word_swap(self, words: List[str], index: int) -> List[str]:
        """Swap two adjacent words."""
        if index < len(words) - 1:
            words[index], words[index + 1] = words[index + 1], words[index]
        return words
    
    def _sentence_restructure(self, words: List[str], index: int) -> List[str]:
        """Restructure sentence order."""
        if len(words) > 6:
            # Split into two parts and swap
            mid = len(words) // 2
            words = words[mid:] + words[:mid]
        return words
    
    def crossover_prompts(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """
        Perform crossover between two prompts.
        
        Args:
            parent1: First parent prompt
            parent2: Second parent prompt
            
        Returns:
            Tuple of two offspring prompts
        """
        words1 = parent1.split()
        words2 = parent2.split()
        
        if not words1 or not words2:
            return parent1, parent2
        
        # Single-point crossover
        point1 = random.randint(1, len(words1) - 1)
        point2 = random.randint(1, len(words2) - 1)
        
        offspring1 = words1[:point1] + words2[point2:]
        offspring2 = words2[:point2] + words1[point1:]
        
        return ' '.join(offspring1), ' '.join(offspring2)
    
    def simple_refinement(self, target_embedding: np.ndarray,
                         config: RefinementConfig) -> PromptCandidate:
        """
        Simple refinement strategy: generate multiple candidates and select best.
        
        Args:
            target_embedding: Target embedding
            config: Refinement configuration
            
        Returns:
            Best prompt candidate
        """
        logger.info("Starting simple refinement")
        
        # Generate initial candidates
        candidates = self.generate_initial_candidates(target_embedding, config.population_size)
        
        # Select best candidate
        best_candidate = max(candidates, key=lambda x: x.score)
        
        logger.info(f"Simple refinement completed. Best score: {best_candidate.score:.4f}")
        return best_candidate
    
    def hill_climbing_refinement(self, target_embedding: np.ndarray,
                                config: RefinementConfig) -> PromptCandidate:
        """
        Hill climbing refinement strategy.
        
        Args:
            target_embedding: Target embedding
            config: Refinement configuration
            
        Returns:
            Best prompt candidate
        """
        logger.info("Starting hill climbing refinement")
        
        # Generate initial candidate
        current_candidate = self.generate_initial_candidates(target_embedding, 1)[0]
        best_candidate = copy.deepcopy(current_candidate)
        
        no_improvement_count = 0
        
        for iteration in range(config.max_iterations):
            # Generate mutated version
            mutated_prompt = self.mutate_prompt(current_candidate.prompt, config.mutation_rate)
            
            # Evaluate mutated prompt
            score, embedding = self.compute_prompt_score(mutated_prompt, target_embedding)
            
            # Accept if better
            if score > current_candidate.score:
                current_candidate = PromptCandidate(
                    prompt=mutated_prompt,
                    score=score,
                    embedding=embedding,
                    metadata={'iteration': iteration}
                )
                
                if score > best_candidate.score:
                    best_candidate = copy.deepcopy(current_candidate)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            logger.info(f"Hill climbing iteration {iteration + 1}: score = {current_candidate.score:.4f}")
            
            # Early stopping
            if no_improvement_count >= config.max_no_improvement:
                logger.info("Early stopping due to no improvement")
                break
        
        logger.info(f"Hill climbing completed. Best score: {best_candidate.score:.4f}")
        return best_candidate
    
    def evolutionary_refinement(self, target_embedding: np.ndarray,
                              config: RefinementConfig) -> PromptCandidate:
        """
        Evolutionary refinement strategy.
        
        Args:
            target_embedding: Target embedding
            config: Refinement configuration
            
        Returns:
            Best prompt candidate
        """
        logger.info("Starting evolutionary refinement")
        
        # Generate initial population
        population = self.generate_initial_candidates(target_embedding, config.population_size)
        best_candidate = max(population, key=lambda x: x.score)
        
        for generation in range(config.max_iterations):
            # Selection (tournament selection)
            new_population = []
            
            for _ in range(config.population_size):
                # Tournament selection
                tournament_size = min(3, len(population))
                tournament = random.sample(population, tournament_size)
                winner = max(tournament, key=lambda x: x.score)
                
                # Crossover
                if random.random() < config.crossover_rate and len(population) > 1:
                    parent2 = random.choice(population)
                    offspring1, offspring2 = self.crossover_prompts(winner.prompt, parent2.prompt)
                    offspring = random.choice([offspring1, offspring2])
                else:
                    offspring = winner.prompt
                
                # Mutation
                if random.random() < config.mutation_rate:
                    offspring = self.mutate_prompt(offspring, config.mutation_rate)
                
                # Evaluate offspring
                score, embedding = self.compute_prompt_score(offspring, target_embedding)
                
                new_population.append(PromptCandidate(
                    prompt=offspring,
                    score=score,
                    embedding=embedding,
                    metadata={'generation': generation}
                ))
            
            # Replace population
            population = new_population
            
            # Update best candidate
            generation_best = max(population, key=lambda x: x.score)
            if generation_best.score > best_candidate.score:
                best_candidate = copy.deepcopy(generation_best)
            
            logger.info(f"Generation {generation + 1}: best score = {best_candidate.score:.4f}")
        
        logger.info(f"Evolutionary refinement completed. Best score: {best_candidate.score:.4f}")
        return best_candidate
    
    def beam_search_refinement(self, target_embedding: np.ndarray,
                             config: RefinementConfig) -> PromptCandidate:
        """
        Beam search refinement strategy.
        
        Args:
            target_embedding: Target embedding
            config: Refinement configuration
            
        Returns:
            Best prompt candidate
        """
        logger.info("Starting beam search refinement")
        
        # Generate initial beam
        beam = self.generate_initial_candidates(target_embedding, config.beam_width)
        best_candidate = max(beam, key=lambda x: x.score)
        
        for iteration in range(config.max_iterations):
            # Expand beam
            expanded_candidates = []
            
            for candidate in beam:
                # Generate mutations
                for _ in range(2):  # 2 mutations per candidate
                    mutated_prompt = self.mutate_prompt(candidate.prompt, config.mutation_rate)
                    score, embedding = self.compute_prompt_score(mutated_prompt, target_embedding)
                    
                    expanded_candidates.append(PromptCandidate(
                        prompt=mutated_prompt,
                        score=score,
                        embedding=embedding,
                        metadata={'iteration': iteration, 'parent': candidate.prompt}
                    ))
            
            # Select top candidates for next beam
            all_candidates = beam + expanded_candidates
            beam = sorted(all_candidates, key=lambda x: x.score, reverse=True)[:config.beam_width]
            
            # Update best candidate
            iteration_best = max(beam, key=lambda x: x.score)
            if iteration_best.score > best_candidate.score:
                best_candidate = copy.deepcopy(iteration_best)
            
            logger.info(f"Beam search iteration {iteration + 1}: best score = {best_candidate.score:.4f}")
        
        logger.info(f"Beam search completed. Best score: {best_candidate.score:.4f}")
        return best_candidate
    
    def simulated_annealing_refinement(self, target_embedding: np.ndarray,
                                     config: RefinementConfig) -> PromptCandidate:
        """
        Simulated annealing refinement strategy.
        
        Args:
            target_embedding: Target embedding
            config: Refinement configuration
            
        Returns:
            Best prompt candidate
        """
        logger.info("Starting simulated annealing refinement")
        
        # Generate initial candidate
        current_candidate = self.generate_initial_candidates(target_embedding, 1)[0]
        best_candidate = copy.deepcopy(current_candidate)
        
        temperature = config.temperature
        
        for iteration in range(config.max_iterations):
            # Generate neighbor
            neighbor_prompt = self.mutate_prompt(current_candidate.prompt, config.mutation_rate)
            score, embedding = self.compute_prompt_score(neighbor_prompt, target_embedding)
            
            neighbor_candidate = PromptCandidate(
                prompt=neighbor_prompt,
                score=score,
                embedding=embedding,
                metadata={'iteration': iteration, 'temperature': temperature}
            )
            
            # Accept or reject
            score_diff = neighbor_candidate.score - current_candidate.score
            
            if score_diff > 0 or random.random() < np.exp(score_diff / temperature):
                current_candidate = neighbor_candidate
                
                if neighbor_candidate.score > best_candidate.score:
                    best_candidate = copy.deepcopy(neighbor_candidate)
            
            # Cool down
            temperature *= config.cooling_rate
            
            logger.info(f"Simulated annealing iteration {iteration + 1}: "
                       f"score = {current_candidate.score:.4f}, temperature = {temperature:.4f}")
        
        logger.info(f"Simulated annealing completed. Best score: {best_candidate.score:.4f}")
        return best_candidate
    
    def refine_prompt(self, target_embedding: np.ndarray,
                     strategy: RefinementStrategy = RefinementStrategy.SIMPLE,
                     config: Optional[RefinementConfig] = None) -> PromptCandidate:
        """
        Refine a prompt using the specified strategy.
        
        Args:
            target_embedding: Target embedding
            strategy: Refinement strategy to use
            config: Refinement configuration
            
        Returns:
            Best refined prompt candidate
        """
        if config is None:
            config = RefinementConfig(strategy=strategy)
        
        logger.info(f"Starting prompt refinement with strategy: {strategy.value}")
        
        start_time = time.time()
        
        # Select refinement strategy
        if strategy == RefinementStrategy.SIMPLE:
            result = self.simple_refinement(target_embedding, config)
        elif strategy == RefinementStrategy.HILL_CLIMBING:
            result = self.hill_climbing_refinement(target_embedding, config)
        elif strategy == RefinementStrategy.EVOLUTIONARY:
            result = self.evolutionary_refinement(target_embedding, config)
        elif strategy == RefinementStrategy.BEAM_SEARCH:
            result = self.beam_search_refinement(target_embedding, config)
        elif strategy == RefinementStrategy.SIMULATED_ANNEALING:
            result = self.simulated_annealing_refinement(target_embedding, config)
        else:
            raise ValueError(f"Unknown refinement strategy: {strategy}")
        
        # Add timing information
        duration = time.time() - start_time
        if result.metadata is None:
            result.metadata = {}
        result.metadata['refinement_time'] = duration
        result.metadata['strategy'] = strategy.value
        
        logger.info(f"Prompt refinement completed in {duration:.2f} seconds")
        return result
    
    def compare_strategies(self, target_embedding: np.ndarray,
                          strategies: List[RefinementStrategy] = None) -> Dict[str, PromptCandidate]:
        """
        Compare multiple refinement strategies.
        
        Args:
            target_embedding: Target embedding
            strategies: List of strategies to compare
            
        Returns:
            Dictionary mapping strategy names to results
        """
        if strategies is None:
            strategies = [
                RefinementStrategy.SIMPLE,
                RefinementStrategy.HILL_CLIMBING,
                RefinementStrategy.EVOLUTIONARY
            ]
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"Running strategy: {strategy.value}")
            result = self.refine_prompt(target_embedding, strategy)
            results[strategy.value] = result
        
        # Log comparison
        logger.info("Strategy comparison results:")
        for strategy, result in results.items():
            logger.info(f"  {strategy}: score = {result.score:.4f}")
        
        return results


def main():
    """Main function for command-line prompt refinement."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Refine prompts using various strategies")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--embedding", required=True, help="Target embedding as JSON string")
    parser.add_argument("--strategy", choices=[s.value for s in RefinementStrategy], 
                       default="simple", help="Refinement strategy")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--population-size", type=int, default=5, help="Population size")
    parser.add_argument("--compare", action="store_true", help="Compare all strategies")
    
    args = parser.parse_args()
    
    # Initialize refinement engine
    refinement = PromptRefinement(args.model_path, args.config)
    
    # Parse embedding
    target_embedding = np.array(json.loads(args.embedding))
    
    # Configure refinement
    config = RefinementConfig(
        strategy=RefinementStrategy(args.strategy),
        max_iterations=args.max_iterations,
        population_size=args.population_size
    )
    
    if args.compare:
        # Compare all strategies
        results = refinement.compare_strategies(target_embedding)
        
        print("\nStrategy Comparison Results:")
        for strategy, result in results.items():
            print(f"{strategy:20s}: {result.score:.4f} - {result.prompt}")
    else:
        # Single strategy
        result = refinement.refine_prompt(target_embedding, RefinementStrategy(args.strategy), config)
        
        print(f"\nRefined prompt: {result.prompt}")
        print(f"Score: {result.score:.4f}")
        print(f"Refinement time: {result.metadata.get('refinement_time', 0):.2f} seconds")


if __name__ == "__main__":
    main() 