import javalang
import re
import sys

class Prepossess:
    def __init__(self, file_path, output_file):
        self.file_path = file_path
        self.output = open(output_file, "w")

    def run(self):
        with open(self.file_path) as f:
            input = f.read()
            split_code = re.findall(r'<s>(.*?)</s>', input)
            for code in split_code:
                self.print_result(code)
                # print("\n")
                self.output.write("\n")


    def print_result(self, code):
        try:
            tree = javalang.parse.parse(code)
            for path, node in tree:
                if node.position is not None:
                    self.print_ast(node)
        except javalang.parser.JavaSyntaxError as e:
            print("Error parsing Java code:")
            print(code)
            print("Exception message:", e)

    def print_ast(self, node, visited=set(), depth=0, ):
        indent = "  " * depth
        if id(node) in visited:
            return
        visited.add(id(node))
        # print(f"{indent}- {type(node).__name__}: {node}\n")
        self.output.write(f"{indent}- {type(node).__name__}: {node}\n")

        if node is not None:
            if hasattr(node, "body") and node.body is not None:
                for i in node.body:
                    self.print_ast(i, visited, depth+1)
                node.body = [i for i in node.body if id(i) not in visited]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py arg1 arg2")
        sys.exit(1)
    sys.setrecursionlimit(5000)
    pre = Prepossess(sys.argv[1], sys.argv[2])
    pre.run()

