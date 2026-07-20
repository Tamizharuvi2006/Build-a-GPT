"""Script entry point for text generation."""
import argparse
import sys
from app.story_generator import StoryGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate text using FantasyLLM")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, help="Prompt (can use stdin)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    args = parser.parse_args()
    
    prompt = args.prompt
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            print("Error: Provide --prompt", file=sys.stderr)
            sys.exit(1)
            
    generator = StoryGenerator(checkpoint_path=args.checkpoint)
    result = generator.generate(
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    print(result)

if __name__ == "__main__":
    main()
