## README

- To run the program, change the working directory to folder $\texttt{src}$  and use command $\texttt{python main.py}$.
- The file $\texttt{Test.py}$ in folder $\texttt{src}$ and two $\texttt{jsonl}$ file in folder $\texttt{data}$ are used for a rough evaluation, you can just ignore them.
- The Generated Embeddings are stored in $\texttt{Embeddings.pkl}$ in folder $\texttt{data}$. It can be loaded via package pickle as a dictionary, where the key is node and the value is its embedding.
- The time of generating embeddings is stored in $\texttt{time.txt}$ of folder $\texttt{data}$, and the result of prediction is stored in file $\texttt{submission.csv}$ in folder $\texttt{data}$.
- The labels are predicted via the cosine similarity of the embeddings of two nodes.

