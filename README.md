TL;DR: Implement papers from scratch to test yourself.

Each folder is:

- Paper name
  - paper_pdf
  - CLAUDE.md
  - test_file
 
`CLAUDE.md` exists to help make the tests that exist in test_file via the PDF which will get passed to claude as an input. 

Then (with good self-regulation), work through the test file and implement the tests. Personally I recommend making copies that way you can go back and revisit the tests.

Rules:

1. No `transformers` or `nn.X` implementations of layers if relevent (e.g. you can't use torch's SDPA while you're writing out SDPA)
2. You may chat with ChatGPT to help you understand the paper better (I recommend this!) but you can't have it give you answers. IMO something like "give me the torch equivalent to doing softmax with XYZ is fine"
3. `pytorch` and/or `einops` is fine

To run the tests, use `pytest`
