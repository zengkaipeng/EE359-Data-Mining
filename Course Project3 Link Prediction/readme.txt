1)To run the program, change the working directory to folder 'src' and use command 'python main.py'
2)The file "Test.py" in folder 'src' and two 'jsonl' file in folder 'data' are used for a rough evaluation, you can just ignore them.
3)The Generated Embeddings are stored in Embeddings.pkl in foler 'data'. It can be loaded via package pickle as a dictionary, where the key is node and the value is its embedding
4)The time of generating embeddings is stored in 'time.txt' of folder 'data', and the result of prediction is stored in file 'submission.csv' in folder 'data'
5)The labels are predicted via the consine similarity of the embeddings of two nodes.