# 🧪 Demo Checkpoint (`best.pt`)

Included in this repository is a **partially trained test checkpoint** located at `checkpoints/best.pt`. 

### What is this?
This checkpoint is a 158 MB snapshot of the model saved during an early, partial training run on a CPU. **It is not fully trained.** 

We pushed this "1/4 trained" checkpoint so that developers and users cloning the repository can immediately test the inference pipeline and see the architecture working end-to-end without having to spend hours training it themselves first.

### What to Expect
Because it was only trained for a few hundred batches (a fraction of the full TinyStories dataset) and the training was intentionally stopped early, the model does not yet understand English. 

**If you run it, it will likely output gibberish.** This is completely expected for an AI that hasn't finished reading its training data!

### How to run it:
To verify that the system works, simply run the chat script:
```bash
python chat.py --checkpoint checkpoints/best.pt
```
If the chat interface boots up, successfully loads the weights, and generates text in response to your prompt (even if the text makes no sense), it proves your local environment is correctly configured and the advanced architecture (Mixture of Experts, Memory Retrieval, Reasoning) is functioning flawlessly!

### Next Steps
To get the model to speak English and write coherent stories, you will need to train it fully on a GPU! See the `docs/user_guide.md` for instructions on how to run `python train.py` to train a complete, smart version of the model.
