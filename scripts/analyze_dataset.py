#!/usr/bin/env python3
"""
Dataset Analysis Script for Alpaca Dataset
Analyzes the prompt-output pairs to understand data distribution and characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
from typing import Dict, List, Any
from collections import Counter
import re
import argparse

# Optional imports
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the alpaca dataset."""
    print(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def basic_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic statistics about the dataset."""
    stats = {
        'total_samples': len(df),
        'null_prompts': df['prompt'].isnull().sum(),
        'null_outputs': df['output'].isnull().sum(),
        'empty_prompts': (df['prompt'] == '').sum(),
        'empty_outputs': (df['output'] == '').sum(),
        'unique_prompts': df['prompt'].nunique(),
        'unique_outputs': df['output'].nunique(),
        'duplicate_prompts': len(df) - df['prompt'].nunique(),
        'duplicate_outputs': len(df) - df['output'].nunique(),
    }
    
    # Text length statistics
    df['prompt_length'] = df['prompt'].str.len()
    df['output_length'] = df['output'].str.len()
    df['prompt_words'] = df['prompt'].str.split().str.len()
    df['output_words'] = df['output'].str.split().str.len()
    
    stats.update({
        'avg_prompt_length': df['prompt_length'].mean(),
        'avg_output_length': df['output_length'].mean(),
        'avg_prompt_words': df['prompt_words'].mean(),
        'avg_output_words': df['output_words'].mean(),
        'min_prompt_length': df['prompt_length'].min(),
        'max_prompt_length': df['prompt_length'].max(),
        'min_output_length': df['output_length'].min(),
        'max_output_length': df['output_length'].max(),
    })
    
    return stats, df


def analyze_prompt_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze different types of prompts."""
    
    # Common prompt starters
    prompt_starters = []
    for prompt in df['prompt'].str.lower():
        if pd.notna(prompt):
            first_word = prompt.split()[0] if prompt.split() else ""
            prompt_starters.append(first_word)
    
    starter_counts = Counter(prompt_starters)
    
    # Identify question vs instruction prompts
    question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'can', 'do', 'is', 'are', 'will', 'would', 'should']
    instruction_words = ['create', 'write', 'describe', 'explain', 'generate', 'make', 'provide', 'give', 'list', 'tell']
    
    questions = df['prompt'].str.lower().str.startswith(tuple(question_words)).sum()
    instructions = df['prompt'].str.lower().str.startswith(tuple(instruction_words)).sum()
    
    # Identify prompts with context/input
    has_context = df['prompt'].str.contains(':', na=False).sum()
    
    return {
        'top_prompt_starters': dict(starter_counts.most_common(20)),
        'question_prompts': questions,
        'instruction_prompts': instructions,
        'context_prompts': has_context,
        'other_prompts': len(df) - questions - instructions
    }


def analyze_output_characteristics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze output characteristics."""
    
    # Count different output types
    code_outputs = df['output'].str.contains('```|def |class |import |from |function', na=False).sum()
    list_outputs = df['output'].str.contains(r'^\d+\.|^-|^\*', na=False).sum()
    explanation_outputs = df['output'].str.contains('because|therefore|thus|however|moreover', na=False).sum()
    
    # Average sentences per output
    df['output_sentences'] = df['output'].str.split(r'[.!?]+').str.len()
    
    return {
        'code_outputs': code_outputs,
        'list_outputs': list_outputs,
        'explanation_outputs': explanation_outputs,
        'avg_sentences_per_output': df['output_sentences'].mean(),
        'median_sentences_per_output': df['output_sentences'].median(),
    }


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualizations for the dataset analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Length distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Text Length Distributions', fontsize=16)
    
    axes[0, 0].hist(df['prompt_length'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Prompt Length Distribution')
    axes[0, 0].set_xlabel('Character Count')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(df['output_length'], bins=50, alpha=0.7, color='red')
    axes[0, 1].set_title('Output Length Distribution')
    axes[0, 1].set_xlabel('Character Count')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[1, 0].hist(df['prompt_words'], bins=50, alpha=0.7, color='green')
    axes[1, 0].set_title('Prompt Word Count Distribution')
    axes[1, 0].set_xlabel('Word Count')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(df['output_words'], bins=50, alpha=0.7, color='orange')
    axes[1, 1].set_title('Output Word Count Distribution')
    axes[1, 1].set_xlabel('Word Count')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_path / 'length_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot of prompt vs output lengths
    plt.figure(figsize=(10, 8))
    plt.scatter(df['prompt_length'], df['output_length'], alpha=0.5, s=10)
    plt.xlabel('Prompt Length (characters)')
    plt.ylabel('Output Length (characters)')
    plt.title('Prompt Length vs Output Length')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'prompt_output_length_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create word cloud for prompts (if available)
    if HAS_WORDCLOUD:
        prompt_text = ' '.join(df['prompt'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(prompt_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Prompts')
        plt.savefig(output_path / 'prompt_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("WordCloud library not available, skipping word cloud generation")
    
    print(f"Visualizations saved to {output_path}")


def save_analysis_results(stats: Dict[str, Any], prompt_analysis: Dict[str, Any], 
                         output_analysis: Dict[str, Any], output_file: str):
    """Save analysis results to a JSON file."""
    
    results = {
        'basic_statistics': stats,
        'prompt_analysis': prompt_analysis,
        'output_analysis': output_analysis,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Analysis results saved to {output_file}")


def print_summary(stats: Dict[str, Any], prompt_analysis: Dict[str, Any], 
                  output_analysis: Dict[str, Any]):
    """Print a summary of the analysis."""
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Basic Statistics:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Unique prompts: {stats['unique_prompts']:,}")
    print(f"  Unique outputs: {stats['unique_outputs']:,}")
    print(f"  Duplicate prompts: {stats['duplicate_prompts']:,}")
    print(f"  Duplicate outputs: {stats['duplicate_outputs']:,}")
    
    print(f"\n📏 Length Statistics:")
    print(f"  Average prompt length: {stats['avg_prompt_length']:.1f} characters")
    print(f"  Average output length: {stats['avg_output_length']:.1f} characters")
    print(f"  Average prompt words: {stats['avg_prompt_words']:.1f}")
    print(f"  Average output words: {stats['avg_output_words']:.1f}")
    
    print(f"\n❓ Prompt Types:")
    print(f"  Question prompts: {prompt_analysis['question_prompts']:,}")
    print(f"  Instruction prompts: {prompt_analysis['instruction_prompts']:,}")
    print(f"  Context prompts: {prompt_analysis['context_prompts']:,}")
    
    print(f"\n💬 Output Characteristics:")
    print(f"  Code outputs: {output_analysis['code_outputs']:,}")
    print(f"  List outputs: {output_analysis['list_outputs']:,}")
    print(f"  Explanation outputs: {output_analysis['explanation_outputs']:,}")
    print(f"  Average sentences per output: {output_analysis['avg_sentences_per_output']:.1f}")
    
    print(f"\n🔤 Top Prompt Starters:")
    for starter, count in list(prompt_analysis['top_prompt_starters'].items())[:10]:
        print(f"  '{starter}': {count:,}")


def main():
    """Main function to run the dataset analysis."""
    
    parser = argparse.ArgumentParser(description="Analyze the Alpaca dataset")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--output-dir", default="analysis_results", help="Output directory for results")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    dataset_path = config['data']['dataset_path']
    
    # Load and analyze dataset
    df = load_dataset(dataset_path)
    stats, df = basic_statistics(df)
    prompt_analysis = analyze_prompt_types(df)
    output_analysis = analyze_output_characteristics(df)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save results
    save_analysis_results(stats, prompt_analysis, output_analysis, 
                         output_dir / "dataset_analysis.json")
    
    # Create visualizations
    if not args.no_viz:
        create_visualizations(df, args.output_dir)
    
    # Print summary
    print_summary(stats, prompt_analysis, output_analysis)
    
    # Save a sample of the data for inspection
    sample_df = df.head(100)
    sample_df.to_csv(output_dir / "dataset_sample.csv", index=False)
    print(f"\nSample of 100 rows saved to {output_dir / 'dataset_sample.csv'}")


if __name__ == "__main__":
    main() 