import javalang
from typing import List
import numpy as np
from gensim.models import Word2Vec
from io import StringIO

class Prepossess:
    def __init__(self, input, line):
        self.input = input
        self.output = ""
        self.line = line

    def run(self):
        # for code in self.input:
        self.output = ""
        self.print_result(self.input)
        return self.output


    def print_result(self, code):
        try:
            tree = javalang.parse.parse(code)
            for path, node in tree:
                if node.position is not None:
                    self.print_ast(node)

        except javalang.parser.JavaSyntaxError as e:
            self.output = "Error parsing Java code"

    def print_ast(self, node, visited=None, depth=0):
        if visited is None:
            visited = set()
        indent = "  " * depth
        if id(node) in visited:
            return
        visited.add(id(node))
        if hasattr(node, "position") and node.position is not None:
            # self.output.write(f"{indent}{type(node).__name__}\n")
            # print(f"{indent}{type(node).__name__}\n")
            position = node.position
            if position.line <= self.line:
                self.output += f"{indent}{type(node).__name__}\n"
        # else:
        #     self.output.write(f"{indent}{type(node).__name__}\n")

        if isinstance(node, javalang.ast.Node):
            # if the node is a container type, iterate over its children and call print_ast() recursively for each child
            for child_name, child_node in node.filter(javalang.ast.Node):
                self.print_ast(child_node, visited, depth + 1)

            # if the node has a body attribute, call print_ast() recursively for each element of the body
            if hasattr(node, "body") and node.body is not None:
                for i in node.body:
                    self.print_ast(i, visited, depth + 1)
                node.body = [i for i in node.body if id(i) not in visited]


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
    def __init__(self, input, vector_size=32):
        self.input = input
        self.vector_size = vector_size

    # Parse the file to get the batches and corpus
    def parseFile(self):
        batches = []
        corpus = []
        lines = self.input.split("\n")
        temp = []
        temp_corpus = []
        for line in lines:
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
                temp_corpus = []
        if temp:
            batches.append(temp)
            corpus.append(temp_corpus)
        return (batches, corpus)

    # A helper function to find the nearest parent of a node in a batch
    def find_nearest_parent(self, batch, curr_idx):
        for i in range(curr_idx, -1, -1):
            if batch[i][1] == batch[curr_idx][1] - 1:
                return i
        # raise IndexError(f"Parent of {curr_idx} not found in list")
        return None

    # Construct a tree from a batch

    def constructTree(self, batch: List[List]) -> TreeNode:
        batch = [["root", 0]] + batch
        batch_nodes = [TreeNode(x[0]) for x in batch]
        for i in range(1, len(batch)):
            parent_idx = self.find_nearest_parent(batch, i)
            if parent_idx is not None:
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
            if len(ret[i].shape) == 2:  # ensure array is 2D
                num_rows = ret[i].shape[0]
                if num_rows < max_rows:
                    padding = ((0, max_rows - num_rows), (0, 0))
                    if ret[i].shape == (2, 2):  # handle (2, 2) case
                        padding = ((0, max_rows - num_rows - 1), (0, 0))
                    ret[i] = np.pad(ret[i], padding, mode="constant")
        ret = np.array(ret, dtype=object)
        flag = True
        enc = ""
        count = 0
        for i in range(ret.shape[0]):
            count += 1
            if count > 1:
                break
            if ret[i].shape[0] == 0 or not flag:
                flag = False
                continue

            # create a StringIO object to hold the formatted array
            f = StringIO()

            # write the formatted array to the StringIO object
            np.savetxt(f, ret[i], fmt="%.3f")

            # get the contents of the StringIO object as a string
            formatted_str = f.getvalue()

            enc += formatted_str
            enc += "\n"

        return enc

if __name__ == "__main__":
    editor1_content = """
    class temp {    public static String encrypt(String plaintext) {
        MessageDigest md = null;
        try {
            md = MessageDigest.getInstance("SHA-1");
        } catch (NoSuchAlgorithmException e) {
            logger.error("NoSuchAlgorithmException:" + e);
        }
        try {
            md.update(plaintext.getBytes("UTF-8"));
        } catch (UnsupportedEncodingException e) {
            logger.error("UnsupportedEncodingException:" + e);
        }
        byte raw[] = md.digest();
        String hash = (new BASE64Encoder()).encode(raw);
        return hash;
    }
} 
    """
    pre1 = Prepossess(editor1_content,100)
    res1 = pre1.run().strip()
    print(res1)
    enc1 = Encoding(res1 + "=\n")
    encoding = enc1.run()
    print(encoding)