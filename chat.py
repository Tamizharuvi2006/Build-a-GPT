"""Simple interactive chat script."""
import argparse
from app.story_generator import StoryGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    print("Loading model...")
    generator = StoryGenerator(checkpoint_path=args.checkpoint)
    
    print("Ready. Type 'quit' to exit.")
    while True:
        try:
            prompt = input("> ")
            if prompt.strip().lower() in ["quit", "exit"]:
                break
            if not prompt.strip():
                continue
            
            response = generator.generate(prompt)
            print(response)
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
