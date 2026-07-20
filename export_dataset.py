from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

with open("tinystories_100mb.txt", "w", encoding="utf-8") as f:
    size = 0
    limit = 100 * 1024 * 1024  # 100 MB

    for story in ds["train"]:
        text = story["text"] + "\n\n"
        encoded = text.encode("utf-8")

        if size + len(encoded) > limit:
            break

        f.write(text)
        size += len(encoded)

print(f"Saved {size / (1024 * 1024):.2f} MB")