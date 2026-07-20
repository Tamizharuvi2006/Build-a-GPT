"""Command-line interface for FantasyData app."""
import argparse
from .story_generator import StoryGenerator

def main():
    parser = argparse.ArgumentParser(description="FantasyData LLM CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    gen_parser = subparsers.add_parser("generate", help="One-shot generation")
    gen_parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    gen_parser.add_argument("--prompt", required=True, type=str, help="Prompt text")
    gen_parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    gen_parser.add_argument("--top-k", type=int, default=50, help="Top-K")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p")
    
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generator = StoryGenerator(checkpoint_path=args.checkpoint)
        response = generator.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(response)
        
    elif args.command == "chat":
        generator = StoryGenerator(checkpoint_path=args.checkpoint)
        print("Starting chat. Type 'quit' to exit.")
        while True:
            try:
                user_input = input("> ")
                if user_input.strip().lower() in ["quit", "exit"]:
                    break
                if not user_input.strip():
                    continue
                response = generator.generate(prompt=user_input)
                print(response)
            except KeyboardInterrupt:
                break
                
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
