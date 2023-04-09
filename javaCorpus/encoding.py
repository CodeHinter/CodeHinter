"""
To run this file, you need to install gensim
pip install gensim
Command to run: python encoding.py output_demo.txt
"""

import sys
from typing import List
import numpy as np
from gensim.models import Word2Vec


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.parent = None
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def depth(self):
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def position(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.children.index(self)


class Encoding:
    def __init__(self, filename, vector_size=128):
        self.filename = filename
        # vector_size for node encoding and position encoding
        # total encoding size = 2 * vector_size
        self.vector_size = vector_size

    # Parse the file to get the batches and corpus
    # example batch: [['FieldDeclaration', 1], ['VariableDeclarationFragment', 2]]
    # example corpus: [['FieldDeclaration', 'VariableDeclarationFragment']]
    def parseFile(self):
        batches = []
        corpus = []
        with open(self.filename, "r") as f:
            temp = []
            temp_corpus = []
            for line in f:
                if "=" not in line:
                    # parse the line to get its depth
                    # e.g. '  FieldDeclaration\n' -> ['FieldDeclaration', 2)]
                    temp.append(
                        [line.strip(), (len(line) - len(line.lstrip())) // 2 + 1]
                    )
                    temp_corpus.append(line.strip())
                else:
                    batches.append(temp)
                    corpus.append(temp_corpus)
                    temp = []
        return (batches, corpus)

    # A helper function to find the nearest parent of a node in a batch
    def find_nearest_parent(self, batch, curr_idx):
        for i in range(curr_idx, -1, -1):
            if batch[i][1] == batch[curr_idx][1] - 1:
                return i
        raise IndexError(f"Parent of {curr_idx} not found in list")

    # Construct a tree from a batch

    def constructTree(self, batch: List[List]) -> TreeNode:
        batch = [["root", 0]] + batch
        batch_nodes = [TreeNode(x[0]) for x in batch]
        for i in range(1, len(batch)):
            parent_idx = self.find_nearest_parent(batch, i)
            batch_nodes[parent_idx].add_child(batch_nodes[i])
        return batch_nodes[0]

    def node2vec(self, corpus):
        # Train the Word2Vec model
        model = Word2Vec(
            corpus, vector_size=self.vector_size, window=5, min_count=1, workers=4
        )
        # Print the most similar words to a given word
        return model

    # position encoding for a single node
    def encode_node(self, node):
        encoding = [node.depth(), node.position()]
        return encoding

    def encode_tree(self, root, batch, node_enc):
        encodings = []
        stack = [(root, [])]
        while stack:
            node, path = stack.pop(0)
            encodings.append(path + self.encode_node(node))
            for child in node.children:
                stack.append((child, path + self.encode_node(node)))
        encodings = encodings[1:]  # remove the root node

        # pad position encoding & add the node encoding to the encodings
        for i in range(len(encodings)):
            if len(encodings[i]) > self.vector_size:
                encodings[i] = encodings[i][: self.vector_size]
            else:
                padding = [0] * (self.vector_size - len(encodings[i]))
                encodings[i].extend(padding)
            encodings[i] += node_enc.wv[batch[i][0]].tolist()
        # print(np.array(encodings).shape)
        return np.array(encodings)

    def run(self):
        batches, corpus = self.parseFile()
        node_enc = self.node2vec(corpus)
        ret = []
        for batch in batches:
            tree = self.constructTree(batch)
            ret.append(self.encode_tree(tree, batch, node_enc))

        # padding each batch to the same shape``
        max_rows = max(arr.shape[0] for arr in ret)
        for i in range(len(ret)):
            num_rows = ret[i].shape[0]
            if num_rows < max_rows:
                padding = ((0, max_rows - num_rows), (0, 0))
                ret[i] = np.pad(ret[i], padding, mode="constant")
        ret = np.array(ret)
        # save the array to a text file with custom formatting
        with open("enc.txt", "w") as f:
            for i in range(ret.shape[0]):
                np.savetxt(f, ret[i], fmt="%.3f")
                f.write("\n")


if __name__ == "__main__":
    enc = Encoding(sys.argv[1])
    enc.run()
